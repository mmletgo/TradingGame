"""Agent 基类模块

本模块定义 Agent 基类，是所有 AI Agent（散户、庄家、做市商）的父类。
"""

from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np

from src.config.config import AgentConfig, AgentType

if TYPE_CHECKING:
    from src.market.matching.matching_engine import MatchingEngine
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

    三种类型 AI Agent（散户、庄家、做市商）的基类，提供通用的属性和方法。

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
    is_liquidated: bool
    _order_counter: int

    def __init__(
        self,
        agent_id: int,
        agent_type: AgentType,
        brain: Brain,
        config: AgentConfig,
    ) -> None:
        """创建 Agent

        初始化 ID、类型、神经网络、账户。

        Args:
            agent_id: Agent ID
            agent_type: Agent 类型
            brain: NEAT 神经网络
            config: Agent 配置
        """
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.brain = brain
        self.account = Account(agent_id, agent_type, config)

        # 预分配神经网络输入缓冲区（607 = 200 + 200 + 100 + 100 + 4 + 3）
        self._input_buffer: np.ndarray = np.zeros(607, dtype=np.float64)

        # 预分配持仓信息缓冲区（4 个值）
        self._position_buffer: np.ndarray = np.zeros(4, dtype=np.float64)

        # 预分配挂单信息缓冲区（3 个值）
        self._pending_order_buffer: np.ndarray = np.zeros(3, dtype=np.float64)

        # 初始化强平标志
        self.is_liquidated = False

        # 初始化订单计数器
        self._order_counter = 0

    def observe(self, market_state: NormalizedMarketState, orderbook: OrderBook) -> np.ndarray:
        """从预计算的市场状态构建神经网络输入

        Args:
            market_state: 预计算的归一化市场数据
            orderbook: 订单簿（用于查询挂单信息）

        Returns:
            神经网络输入向量（ndarray）
        """
        # 直接复制到预分配数组，避免 np.concatenate 创建新数组
        self._input_buffer[:200] = market_state.bid_data
        self._input_buffer[200:400] = market_state.ask_data
        self._input_buffer[400:500] = market_state.trade_prices
        self._input_buffer[500:600] = market_state.trade_quantities
        self._input_buffer[600:604] = self._get_position_inputs(market_state.mid_price)
        self._input_buffer[604:607] = self._get_pending_order_inputs(market_state.mid_price, orderbook)
        return self._input_buffer  # 不调用 .tolist()

    def _get_position_inputs(self, mid_price: float) -> np.ndarray:
        """获取持仓信息输入（4 个值）

        Args:
            mid_price: 中间价

        Returns:
            持仓信息输入向量 [持仓归一化, 均价归一化, 余额归一化, 净值归一化]
        """
        # 使用预分配的缓冲区避免每次创建新数组
        result = self._position_buffer

        # 持仓价值归一化
        equity = self.account.get_equity(mid_price)
        position_value = abs(self.account.position.quantity) * mid_price
        if equity > 0 and self.account.leverage > 0:
            result[0] = position_value / (equity * self.account.leverage)
        else:
            result[0] = 0.0

        # 持仓均价归一化
        if self.account.position.quantity == 0:
            result[1] = 0.0
        else:
            result[1] = (self.account.position.avg_price - mid_price) / mid_price if mid_price > 0 else 0.0

        # 余额归一化
        initial_balance = self.account.initial_balance
        result[2] = self.account.balance / initial_balance if initial_balance > 0 else 0.0

        # 净值归一化
        result[3] = equity / initial_balance if initial_balance > 0 else 0.0

        return result

    def _get_pending_order_inputs(self, mid_price: float, orderbook: OrderBook) -> np.ndarray:
        """获取挂单信息输入（3 个值：价格归一化、数量、方向）

        子类可重写此方法以支持不同的挂单格式。

        Args:
            mid_price: 中间价
            orderbook: 订单簿

        Returns:
            挂单信息输入向量 [价格归一化, 数量, 方向]
        """
        # 使用预分配的缓冲区避免每次创建新数组
        result = self._pending_order_buffer

        pending_id = self.account.pending_order_id
        if pending_id is None:
            result[0] = 0.0
            result[1] = 0.0
            result[2] = 0.0
            return result

        order = orderbook.order_map.get(pending_id)
        if order is None:
            result[0] = 0.0
            result[1] = 0.0
            result[2] = 0.0
            return result

        result[0] = (order.price - mid_price) / mid_price if mid_price > 0 else 0.0
        result[1] = float(order.quantity)
        result[2] = 1.0 if order.side == OrderSide.BUY else -1.0
        return result

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
        # 如果已被强平，直接返回 HOLD
        if self.is_liquidated:
            return ActionType.HOLD, {}

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
            raw_price = mid_price + price_offset_ticks * tick_size
            # 舍入到 tick_size 的整数倍，避免浮点数精度问题
            # 确保价格至少为一个 tick_size，防止出现负价格或零价格
            params["price"] = max(tick_size, round(raw_price / tick_size) * tick_size)
            # 数量由神经网络决定
            params["quantity"] = self._calculate_order_quantity(mid_price, quantity_ratio)

        elif action == ActionType.PLACE_ASK:
            # 挂卖单：价格由神经网络决定（相对 mid_price 的偏移）
            price_offset_ticks = price_offset_norm * 100  # ±100 ticks
            raw_price = mid_price + price_offset_ticks * tick_size
            # 舍入到 tick_size 的整数倍，避免浮点数精度问题
            # 确保价格至少为一个 tick_size，防止出现负价格或零价格
            params["price"] = max(tick_size, round(raw_price / tick_size) * tick_size)
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
                # 取整并确保至少卖出1个单位
                sell_qty = max(1, int(position_qty * quantity_ratio))
                # 但不能超过持仓量
                params["quantity"] = min(sell_qty, int(position_qty))
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

    # 订单数量上限，防止 int 溢出
    MAX_ORDER_QUANTITY: int = 100_000_000

    def _calculate_order_quantity(self, price: float, ratio: float) -> int:
        """计算订单数量

        根据账户净值、杠杆倍数和数量比例计算订单数量。

        Args:
            price: 价格
            ratio: 数量比例（0.1 到 1.0，表示使用购买力的比例）

        Returns:
            订单数量（整数），如果净值为负或不足则返回 0
        """
        equity = self.account.get_equity(price)

        # 净值非正时不允许下单
        if equity <= 0:
            return 0

        # 可用购买力 = 净值 * 杠杆
        buying_power = equity * self.account.leverage
        # 限制比例在合理范围
        ratio = max(0.1, min(1.0, ratio))
        quantity = (buying_power * ratio) / price if price > 0 else 0.0

        # 确保数量为整数且至少为1（最小交易单位），同时限制最大值防止溢出
        quantity = max(1, min(self.MAX_ORDER_QUANTITY, int(quantity)))

        return quantity

    def execute_action(
        self,
        action: ActionType,
        params: dict[str, Any],
        matching_engine: "MatchingEngine",
    ) -> list[Trade]:
        """执行动作

        直接调用撮合引擎处理订单。
        成交后直接更新账户。

        Args:
            action: 动作类型
            params: 动作参数字典
                - PLACE_BID/PLACE_ASK: {"price": float, "quantity": float}
                - MARKET_BUY/MARKET_SELL: {"quantity": float}
                - CANCEL: {"order_id": int} (可选，默认使用账户的 pending_order_id)
                - CLEAR_POSITION/HOLD: {}
            matching_engine: 撮合引擎

        Returns:
            成交列表
        """
        if self.is_liquidated:
            return []

        trades: list[Trade] = []

        if action == ActionType.PLACE_BID:
            trades = self._place_limit_order(
                OrderSide.BUY, params["price"], params["quantity"], matching_engine
            )
        elif action == ActionType.PLACE_ASK:
            trades = self._place_limit_order(
                OrderSide.SELL, params["price"], params["quantity"], matching_engine
            )
        elif action == ActionType.CANCEL:
            order_id = params.get("order_id") or self.account.pending_order_id
            if order_id is not None:
                matching_engine.cancel_order(order_id)
        elif action == ActionType.MARKET_BUY:
            trades = self._place_market_order(
                OrderSide.BUY, params["quantity"], matching_engine
            )
        elif action == ActionType.MARKET_SELL:
            trades = self._place_market_order(
                OrderSide.SELL, params["quantity"], matching_engine
            )
        elif action == ActionType.CLEAR_POSITION:
            trades = self._handle_clear_position(matching_engine)
        # HOLD: 不执行任何操作

        return trades

    def _place_limit_order(
        self,
        side: OrderSide,
        price: float,
        quantity: int,
        matching_engine: "MatchingEngine",
    ) -> list[Trade]:
        """下限价单

        Args:
            side: 订单方向
            price: 价格
            quantity: 数量
            matching_engine: 撮合引擎

        Returns:
            成交列表
        """
        # 数量无效时不下单
        if quantity <= 0:
            return []

        order = Order(
            order_id=self._generate_order_id(),
            agent_id=self.agent_id,
            side=side,
            order_type=OrderType.LIMIT,
            price=price,
            quantity=quantity,
        )
        trades = matching_engine.process_order(order)
        self._process_trades(trades)

        # 更新挂单ID（如果订单未完全成交，则记录）
        if matching_engine._orderbook.order_map.get(order.order_id) is not None:
            self.account.pending_order_id = order.order_id
        else:
            self.account.pending_order_id = None

        return trades

    def _place_market_order(
        self,
        side: OrderSide,
        quantity: int,
        matching_engine: "MatchingEngine",
    ) -> list[Trade]:
        """下市价单

        Args:
            side: 订单方向
            quantity: 数量
            matching_engine: 撮合引擎

        Returns:
            成交列表
        """
        # 数量无效时不下单
        if quantity <= 0:
            return []

        order = Order(
            order_id=self._generate_order_id(),
            agent_id=self.agent_id,
            side=side,
            order_type=OrderType.MARKET,
            price=0.0,
            quantity=quantity,
        )
        trades = matching_engine.process_order(order)
        self._process_trades(trades)
        return trades

    def _handle_clear_position(
        self,
        matching_engine: "MatchingEngine",
    ) -> list[Trade]:
        """处理清仓

        Args:
            matching_engine: 撮合引擎

        Returns:
            成交列表
        """
        position_qty = self.account.position.quantity
        if position_qty > 0:
            return self._place_market_order(OrderSide.SELL, position_qty, matching_engine)
        elif position_qty < 0:
            return self._place_market_order(OrderSide.BUY, abs(position_qty), matching_engine)
        return []

    def _process_trades(self, trades: list[Trade]) -> None:
        """处理成交列表，更新账户

        Args:
            trades: 成交列表
        """
        for trade in trades:
            # 使用 is_buyer_taker 判断 taker 是买方还是卖方
            # 旧逻辑 `is_buyer = trade.buyer_id == self.agent_id` 在自成交时会出错
            # 因为自成交时 buyer_id == seller_id == self.agent_id
            is_buyer = trade.is_buyer_taker
            self.account.on_trade(trade, is_buyer)

    def _generate_order_id(self) -> int:
        """生成唯一订单ID

        使用 agent_id 和递增计数器组合，确保多 Agent 时的唯一性。
        比 MD5 哈希更高效，适合高频交易场景。

        Returns:
            唯一的订单ID（正整数）
        """
        self._order_counter += 1
        # 组合 agent_id 和计数器：agent_id 占高 32 位，计数器占低 32 位
        return (self.agent_id << 32) | self._order_counter

    def reset(self, config: AgentConfig) -> None:
        """重置 Agent 状态

        重置账户余额、持仓、挂单，用于恢复训练或重置演示。

        Args:
            config: 新的 Agent 配置对象，用于初始化账户
        """
        # 重置账户状态（余额、持仓、挂单ID、杠杆、费率等）
        self.account = Account(self.agent_id, self.agent_type, config)

        # 重置强平标志
        self.is_liquidated = False

        # 重置订单计数器
        self._order_counter = 0
