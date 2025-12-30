"""Agent 基类模块

本模块定义 Agent 基类，是所有 AI Agent（散户、庄家、做市商）的父类。
"""

import hashlib
import time
from enum import Enum
from typing import Any

from src.config.config import AgentConfig, AgentType
from src.core.event_engine.events import Event, EventType
from src.core.event_engine.event_bus import EventBus
from src.bio.brain.brain import Brain
from src.market.account.account import Account
from src.market.matching.trade import Trade
from src.market.market_state import NormalizedMarketState
from src.market.orderbook.order import Order, OrderSide, OrderType
from src.market.orderbook.orderbook import OrderBook


class ActionType(Enum):
    """动作类型枚举

    定义 Agent 可以执行的所有交易动作。
    """

    HOLD = 0  # 不动
    PLACE_BID = 1  # 挂买单
    PLACE_ASK = 2  # 挂卖单
    CANCEL = 3  # 撤单
    MARKET_BUY = 4  # 市价买入
    MARKET_SELL = 5  # 市价卖出
    CLEAR_POSITION = 6  # 清仓
    QUOTE = 7  # 做市商双边挂单（每边1-10单）


class Agent:
    """Agent 基类

    三种类型 AI Agent（散户、庄家、做市商）的基类，提供通用的属性和事件处理。

    Attributes:
        agent_id: Agent ID
        agent_type: Agent 类型（散户/庄家/做市商）
        brain: NEAT 神经网络
        account: 交易账户
    """

    agent_id: int
    agent_type: AgentType
    brain: Brain
    account: Account

    def __init__(
        self,
        agent_id: int,
        agent_type: AgentType,
        brain: Brain,
        config: AgentConfig,
        event_bus: EventBus,
    ) -> None:
        """创建 Agent

        初始化 ID、类型、神经网络、账户，并订阅成交事件。

        Args:
            agent_id: Agent ID
            agent_type: Agent 类型
            brain: NEAT 神经网络
            config: Agent 配置
            event_bus: 事件总线
        """
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.brain = brain
        self.account = Account(agent_id, agent_type, config)
        self.event_bus = event_bus

        # 使用带 ID 的订阅，支持定向发送
        event_bus.subscribe_with_id(EventType.TRADE_EXECUTED, agent_id, self._on_trade_event)

    def _on_trade_event(self, event: Event) -> None:
        """处理成交事件（已定向发送，无需过滤）

        Args:
            event: 成交事件，包含 trade_id, price, quantity, buyer_id, seller_id,
                   buyer_fee, seller_fee, is_buyer_taker 等字段
        """
        # 从事件数据构建 Trade 对象
        trade = Trade(
            trade_id=event.data["trade_id"],
            price=event.data["price"],
            quantity=event.data["quantity"],
            buyer_id=event.data["buyer_id"],
            seller_id=event.data["seller_id"],
            buyer_fee=event.data["buyer_fee"],
            seller_fee=event.data["seller_fee"],
            is_buyer_taker=event.data["is_buyer_taker"],
            timestamp=event.timestamp,
        )

        is_buyer = event.data["buyer_id"] == self.agent_id
        self.account.on_trade(trade, is_buyer)

    def observe(self, market_state: NormalizedMarketState, orderbook: OrderBook) -> list[float]:
        """从预计算的市场状态构建神经网络输入

        Args:
            market_state: 预计算的归一化市场数据
            orderbook: 订单簿（用于查询挂单信息）

        Returns:
            神经网络输入向量
        """
        inputs: list[float] = []

        # 1. 买盘（直接使用预计算值）- 200 个
        inputs.extend(market_state.bid_data.tolist())

        # 2. 卖盘（直接使用预计算值）- 200 个
        inputs.extend(market_state.ask_data.tolist())

        # 3. 成交（价格 + 数量带方向）- 200 个
        # 数量正负表示方向：正=taker买入，负=taker卖出
        inputs.extend(market_state.trade_prices.tolist())
        inputs.extend(market_state.trade_quantities.tolist())

        # 4. 持仓信息 - 4 个
        inputs.extend(self._get_position_inputs(market_state.mid_price))

        # 5. 挂单信息 - 3 个（子类可重写）
        inputs.extend(self._get_pending_order_inputs(market_state.mid_price, orderbook))

        return inputs

    def _get_position_inputs(self, mid_price: float) -> list[float]:
        """获取持仓信息输入（4 个值）

        Args:
            mid_price: 中间价

        Returns:
            持仓信息输入向量 [持仓归一化, 均价归一化, 余额归一化, 净值归一化]
        """
        inputs: list[float] = []

        # 持仓价值归一化
        equity = self.account.get_equity(mid_price)
        position_value = abs(self.account.position.quantity) * mid_price
        if equity > 0 and self.account.leverage > 0:
            position_norm = position_value / (equity * self.account.leverage)
        else:
            position_norm = 0.0
        inputs.append(position_norm)

        # 持仓均价归一化
        if self.account.position.quantity == 0:
            inputs.append(0.0)
        else:
            avg_price_norm = (self.account.position.avg_price - mid_price) / mid_price if mid_price > 0 else 0
            inputs.append(avg_price_norm)

        # 余额归一化
        initial_balance = self.account.initial_balance
        balance_norm = self.account.balance / initial_balance if initial_balance > 0 else 0
        inputs.append(balance_norm)

        # 净值归一化
        equity_norm = equity / initial_balance if initial_balance > 0 else 0
        inputs.append(equity_norm)

        return inputs

    def _get_pending_order_inputs(self, mid_price: float, orderbook: OrderBook) -> list[float]:
        """获取挂单信息输入（3 个值：价格归一化、数量、方向）

        子类可重写此方法以支持不同的挂单格式。

        Args:
            mid_price: 中间价
            orderbook: 订单簿

        Returns:
            挂单信息输入向量 [价格归一化, 数量, 方向]
        """
        pending_id = self.account.pending_order_id
        if pending_id is None:
            return [0.0, 0.0, 0.0]

        order = orderbook.order_map.get(pending_id)
        if order is None:
            return [0.0, 0.0, 0.0]

        price_norm = (order.price - mid_price) / mid_price if mid_price > 0 else 0
        direction = 1.0 if order.side == OrderSide.BUY else -1.0
        return [price_norm, float(order.quantity), direction]

    def decide(self, market_state: NormalizedMarketState, orderbook: OrderBook) -> tuple[ActionType, dict[str, Any]]:
        """决策下一步动作（接收预计算的市场状态）

        观察市场状态，通过神经网络前向传播，解析输出为动作类型和参数。

        神经网络输出结构：
        - 输出[0-6]: 7 种动作类型的得分（选择最大值）
        - 输出[7]: 价格偏移（归一化值，-1 到 1，相对于 mid_price）
        - 输出[8]: 数量比例（归一化值，0 到 1，相对于可用购买力）

        Args:
            market_state: 预计算的归一化市场数据
            orderbook: 订单簿

        Returns:
            (动作类型, 动作参数字典)
            - PLACE_BID/PLACE_ASK: {"price": float, "quantity": float}
            - MARKET_BUY/MARKET_SELL: {"quantity": float}
            - CANCEL/CLEAR_POSITION/HOLD: {}
        """
        # 1. 观察市场，获取神经网络输入
        inputs = self.observe(market_state, orderbook)

        # 2. 神经网络前向传播
        outputs = self.brain.forward(inputs)

        # 3. 验证输出维度（至少需要 9 个值：7个动作 + 价格偏移 + 数量比例）
        if len(outputs) < 9:
            raise ValueError(f"神经网络输出维度不足，期望 9，实际 {len(outputs)}")

        # 4. 解析动作类型（选择前 7 个输出中值最大的索引）
        action_idx = int(max(range(7), key=lambda i: outputs[i]))
        action = ActionType(action_idx)

        # 5. 解析参数（由神经网络决定）
        # 输出[7]: 价格偏移（-1 到 1，映射到 ±100 个 tick）
        # 输出[8]: 数量比例（-1 到 1，映射到 0.1-1.0 的购买力比例）
        price_offset_norm = max(-1.0, min(1.0, outputs[7]))  # 限制在 [-1, 1]
        quantity_ratio_norm = max(-1.0, min(1.0, outputs[8]))  # 限制在 [-1, 1]

        # 获取参考价格
        mid_price = market_state.mid_price
        if mid_price == 0:
            mid_price = 100.0

        # 映射数量比例到 [0.1, 1.0]
        quantity_ratio = 0.1 + (quantity_ratio_norm + 1) * 0.45  # -1→0.1, 0→0.55, 1→1.0

        # 获取 tick_size
        tick_size = market_state.tick_size if market_state.tick_size > 0 else 0.1

        # 根据动作类型计算参数
        params: dict[str, Any] = {}

        if action == ActionType.PLACE_BID:
            # 挂买单：价格由神经网络决定（相对 mid_price 的偏移）
            price_offset_ticks = price_offset_norm * 100  # ±100 ticks
            params["price"] = mid_price + price_offset_ticks * tick_size
            # 数量由神经网络决定
            params["quantity"] = self._calculate_order_quantity(mid_price, quantity_ratio)

        elif action == ActionType.PLACE_ASK:
            # 挂卖单：价格由神经网络决定（相对 mid_price 的偏移）
            price_offset_ticks = price_offset_norm * 100  # ±100 ticks
            params["price"] = mid_price + price_offset_ticks * tick_size
            # 数量由神经网络决定
            params["quantity"] = self._calculate_order_quantity(mid_price, quantity_ratio)

        elif action == ActionType.MARKET_BUY:
            # 市价买入：数量由神经网络决定
            params["quantity"] = self._calculate_order_quantity(mid_price, quantity_ratio)

        elif action == ActionType.MARKET_SELL:
            # 市价卖出：数量由神经网络决定
            position_qty = self.account.position.quantity
            if position_qty > 0:
                # 有多仓时卖出（卖出比例由神经网络决定）
                params["quantity"] = position_qty * quantity_ratio
            else:
                # 空仓或无持仓，开空仓
                params["quantity"] = self._calculate_order_quantity(mid_price, quantity_ratio)

        elif action == ActionType.CANCEL:
            # 撤单：无参数
            pass

        elif action == ActionType.CLEAR_POSITION:
            # 清仓：无参数（由调用方处理）
            pass

        elif action == ActionType.HOLD:
            # 不动：无参数
            pass

        return action, params

    def _calculate_order_quantity(self, price: float, ratio: float) -> float:
        """计算订单数量

        根据账户净值、杠杆倍数和数量比例计算订单数量。

        Args:
            price: 价格
            ratio: 数量比例（0.1 到 1.0，表示使用购买力的比例）

        Returns:
            订单数量
        """
        equity = self.account.get_equity(price)
        # 可用购买力 = 净值 * 杠杆
        buying_power = equity * self.account.leverage
        # 限制比例在合理范围
        ratio = max(0.1, min(1.0, ratio))
        quantity = (buying_power * ratio) / price if price > 0 else 0

        # 确保数量为正数且合理（至少为最小交易单位）
        lot_size = 1.0  # 默认最小交易单位
        quantity = max(lot_size, round(quantity, 2))

        return quantity

    def execute_action(self, action: ActionType, params: dict[str, Any], event_bus: EventBus) -> None:
        """执行动作

        根据动作类型发布订单事件到事件总线，由撮合引擎处理。

        Args:
            action: 动作类型
            params: 动作参数字典
                - PLACE_BID/PLACE_ASK: {"price": float, "quantity": float}
                - MARKET_BUY/MARKET_SELL: {"quantity": float}
                - CANCEL: {"order_id": int} (可选，默认使用账户的 pending_order_id)
                - CLEAR_POSITION/HOLD: {}
            event_bus: 事件总线
        """
        if action == ActionType.HOLD:
            # 不动：不做任何操作
            return

        timestamp = time.time()

        if action == ActionType.PLACE_BID:
            # 挂买单：创建限价买单
            price = params["price"]
            quantity = params["quantity"]
            order_id = self._generate_order_id()
            order = Order(
                order_id=order_id,
                agent_id=self.agent_id,
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                price=price,
                quantity=quantity,
            )
            event = Event(EventType.ORDER_PLACED, timestamp, {"order": order})
            event_bus.publish(event)

        elif action == ActionType.PLACE_ASK:
            # 挂卖单：创建限价卖单
            price = params["price"]
            quantity = params["quantity"]
            order_id = self._generate_order_id()
            order = Order(
                order_id=order_id,
                agent_id=self.agent_id,
                side=OrderSide.SELL,
                order_type=OrderType.LIMIT,
                price=price,
                quantity=quantity,
            )
            event = Event(EventType.ORDER_PLACED, timestamp, {"order": order})
            event_bus.publish(event)

        elif action == ActionType.CANCEL:
            # 撤单：从参数中获取订单ID，或使用账户的 pending_order_id
            order_id = params.get("order_id")
            if order_id is None:
                order_id = self.account.pending_order_id

            if order_id is not None:
                event = Event(
                    EventType.ORDER_CANCELLED,
                    timestamp,
                    {"order_id": order_id, "agent_id": self.agent_id},
                )
                event_bus.publish(event)

        elif action == ActionType.MARKET_BUY:
            # 市价买入：创建市价买单
            quantity = params["quantity"]
            order_id = self._generate_order_id()
            order = Order(
                order_id=order_id,
                agent_id=self.agent_id,
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                price=0.0,  # 市价单价格无意义
                quantity=quantity,
            )
            event = Event(EventType.ORDER_PLACED, timestamp, {"order": order})
            event_bus.publish(event)

        elif action == ActionType.MARKET_SELL:
            # 市价卖出：创建市价卖单
            quantity = params["quantity"]
            order_id = self._generate_order_id()
            order = Order(
                order_id=order_id,
                agent_id=self.agent_id,
                side=OrderSide.SELL,
                order_type=OrderType.MARKET,
                price=0.0,  # 市价单价格无意义
                quantity=quantity,
            )
            event = Event(EventType.ORDER_PLACED, timestamp, {"order": order})
            event_bus.publish(event)

        elif action == ActionType.CLEAR_POSITION:
            # 清仓：根据持仓方向发布市价单
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

    def _generate_order_id(self) -> int:
        """生成唯一订单ID

        使用 agent_id 和纳秒级时间戳进行哈希处理，确保多 Agent 并发时的唯一性。

        Returns:
            唯一的订单ID（正整数）
        """
        # 使用 MD5 哈希 agent_id 和纳秒时间戳的组合
        data = f"{self.agent_id}:{time.time_ns()}".encode()
        hash_bytes = hashlib.md5(data).digest()
        # 取前 8 字节转为整数（确保为正数）
        order_id = int.from_bytes(hash_bytes[:8], byteorder="big", signed=False)
        return order_id

    def get_fitness(self, current_price: float) -> float:
        """计算适应度

        适应度 = 账户净值 / 初始净值，用于 NEAT 进化算法评估 Agent 的交易表现。

        Args:
            current_price: 当前市场价格，用于计算账户净值

        Returns:
            适应度值（>1 表示盈利，=1 表示持平，<1 表示亏损）
        """
        equity = self.account.get_equity(current_price)
        return equity / self.account.initial_balance

    def reset(self, config: AgentConfig) -> None:
        """重置 Agent 状态

        清除事件订阅，重置账户余额、持仓、挂单，用于恢复训练或重置演示。

        Args:
            config: 新的 Agent 配置对象，用于初始化账户
        """
        # 取消带 ID 的订阅
        self.event_bus.unsubscribe_with_id(EventType.TRADE_EXECUTED, self.agent_id)

        # 重置账户状态（余额、持仓、挂单ID、杠杆、费率等）
        self.account = Account(self.agent_id, self.agent_type, config)

        # 重新订阅成交事件
        self.event_bus.subscribe_with_id(EventType.TRADE_EXECUTED, self.agent_id, self._on_trade_event)
