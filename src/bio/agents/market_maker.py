"""做市商 Agent 模块

本模块定义做市商 Agent 类，继承自 Agent 基类。
"""

from typing import TYPE_CHECKING, Any

import numpy as np

from src.bio.agents.base import Agent, ActionType
from src.bio.brain.brain import Brain
from src.config.config import AgentConfig, AgentType
from src.market.market_state import NormalizedMarketState
from src.market.orderbook.order import Order, OrderSide, OrderType
from src.market.orderbook.orderbook import OrderBook
from src.market.matching.trade import Trade

if TYPE_CHECKING:
    from src.market.matching.matching_engine import MatchingEngine


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
        self, agent_id: int, brain: Brain, config: AgentConfig
    ) -> None:
        """创建做市商 Agent

        调用父类构造函数，设置类型为 MARKET_MAKER，初始化买卖挂单列表。

        Args:
            agent_id: Agent ID
            brain: NEAT 神经网络
            config: Agent 配置
        """
        super().__init__(agent_id, AgentType.MARKET_MAKER, brain, config)

        # 做市商每边（买/卖）可以挂1-10个订单
        self.bid_order_ids: list[int] = []
        self.ask_order_ids: list[int] = []

        # 做市商输入更大（634 = 604 + 30 挂单信息）
        self._input_buffer = np.zeros(634, dtype=np.float64)

    def get_action_space(self) -> list[ActionType]:
        """获取做市商可用动作

        做市商可以执行双边挂单、清仓两种动作。

        Returns:
            可用动作类型列表 [QUOTE, CLEAR_POSITION]
        """
        return [ActionType.QUOTE, ActionType.CLEAR_POSITION]

    def observe(
        self, market_state: NormalizedMarketState, orderbook: OrderBook
    ) -> np.ndarray:
        """从预计算的市场状态构建神经网络输入

        做市商覆盖基类方法，使用更大的输入缓冲区（634 = 604 + 30 挂单信息）。

        Args:
            market_state: 预计算的归一化市场数据
            orderbook: 订单簿（用于查询挂单信息）

        Returns:
            神经网络输入向量（ndarray）
        """
        # 直接复制到预分配数组
        self._input_buffer[:200] = market_state.bid_data
        self._input_buffer[200:400] = market_state.ask_data
        self._input_buffer[400:500] = market_state.trade_prices
        self._input_buffer[500:600] = market_state.trade_quantities
        self._input_buffer[600:604] = self._get_position_inputs(market_state.mid_price)
        self._input_buffer[604:634] = self._get_pending_order_inputs(
            market_state.mid_price, orderbook
        )
        return self._input_buffer  # 不调用 .tolist()

    def _fill_order_inputs(
        self,
        inputs: np.ndarray,
        order_ids: list[int],
        offset: int,
        mid_price: float,
        orderbook: OrderBook,
    ) -> None:
        """填充订单输入数组

        Args:
            inputs: 输入数组（将被修改）
            order_ids: 订单ID列表
            offset: 在数组中的起始偏移量
            mid_price: 中间价（用于价格归一化）
            orderbook: 订单簿（用于查询订单详情）
        """
        for i in range(5):
            if i < len(order_ids):
                order = orderbook.order_map.get(order_ids[i])
                if order is not None:
                    price_norm = (
                        (order.price - mid_price) / mid_price if mid_price > 0 else 0.0
                    )
                    idx = offset + i * 3
                    inputs[idx] = price_norm
                    inputs[idx + 1] = float(order.quantity)
                    inputs[idx + 2] = 1.0

    def _get_pending_order_inputs(
        self, mid_price: float, orderbook: OrderBook
    ) -> np.ndarray:
        """获取做市商挂单信息（30 个值）

        买单 5 个位置 x 3 = 15
        卖单 5 个位置 x 3 = 15
        每个位置：[价格归一化, 数量, 有效标志(1.0/0.0)]

        Args:
            mid_price: 中间价（用于价格归一化）
            orderbook: 订单簿（用于查询订单详情）

        Returns:
            30 个浮点数的 NumPy 数组
        """
        inputs = np.zeros(30, dtype=np.float64)
        self._fill_order_inputs(inputs, self.bid_order_ids, 0, mid_price, orderbook)
        self._fill_order_inputs(inputs, self.ask_order_ids, 15, mid_price, orderbook)
        return inputs

    def decide(
        self, market_state: NormalizedMarketState, orderbook: OrderBook
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
            market_state: 预计算的归一化市场数据
            orderbook: 订单簿对象

        Returns:
            (动作类型, 动作参数字典)
            - QUOTE: {"bid_orders": [{"price": float, "quantity": float}, ...],
                      "ask_orders": [{"price": float, "quantity": float}, ...]}
            - CLEAR_POSITION: {}
        """
        # 如果已被强平，返回清仓（实际不执行）
        if self.is_liquidated:
            return ActionType.CLEAR_POSITION, {}

        # 1. 观察市场，获取神经网络输入（复用基类方法）
        inputs = self.observe(market_state, orderbook)

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
        mid_price = market_state.mid_price
        if mid_price == 0:
            mid_price = 100.0

        tick_size = market_state.tick_size if market_state.tick_size > 0 else 0.1

        # 向量化收集数量比例
        outputs_arr = np.array(outputs)
        bid_raw_ratios = np.maximum(0.0, (np.clip(outputs_arr[7:12], -1, 1) + 1) / 2)
        ask_raw_ratios = np.maximum(0.0, (np.clip(outputs_arr[17:22], -1, 1) + 1) / 2)

        # 计算总和并归一化（确保 10 个订单的总比例 = 1.0）
        total_raw_ratio = bid_raw_ratios.sum() + ask_raw_ratios.sum()
        if total_raw_ratio > 0:
            bid_ratios = bid_raw_ratios / total_raw_ratio
            ask_ratios = ask_raw_ratios / total_raw_ratio
        else:
            bid_ratios = np.zeros(5)
            ask_ratios = np.zeros(5)

        # 向量化解析买单价格
        # 每个订单有基础偏移（i+1）* 20 ticks，加上神经网络微调 ±10 ticks
        bid_price_offsets = np.clip(outputs_arr[2:7], -1, 1)
        base_offsets = np.array(
            [20.0, 40.0, 60.0, 80.0, 100.0]
        )  # 基础偏移 2-10 个价格单位
        nn_adjusts = bid_price_offsets * 20.0  # 神经网络微调 ±1 个价格单位
        bid_price_ticks = np.maximum(1.0, base_offsets + nn_adjusts)
        # 舍入到 tick_size 的整数倍，避免浮点数精度问题
        # 确保价格至少为一个 tick_size，防止出现负价格或零价格
        bid_prices = np.maximum(
            tick_size,
            np.round((mid_price - bid_price_ticks * tick_size) / tick_size) * tick_size
        )

        bid_orders: list[dict[str, float]] = []
        # 构建买单列表（仍需循环计算数量，但价格已向量化）
        for i in range(5):
            quantity = self._calculate_order_quantity(
                float(bid_prices[i]), float(bid_ratios[i])
            )
            # 只添加数量 > 0 的订单
            if quantity > 0:
                bid_orders.append({"price": float(bid_prices[i]), "quantity": quantity})

        # 向量化解析卖单价格
        ask_price_offsets = np.clip(outputs_arr[12:17], -1, 1)
        nn_adjusts = ask_price_offsets * 20.0  # 神经网络微调 ±1 个价格单位
        ask_price_ticks = np.maximum(1.0, base_offsets + nn_adjusts)
        # 舍入到 tick_size 的整数倍，避免浮点数精度问题
        ask_prices = (
            np.round((mid_price + ask_price_ticks * tick_size) / tick_size) * tick_size
        )

        ask_orders: list[dict[str, float]] = []
        # 构建卖单列表（仍需循环计算数量，但价格已向量化）
        for i in range(5):
            quantity = self._calculate_order_quantity(
                float(ask_prices[i]), float(ask_ratios[i])
            )
            # 只添加数量 > 0 的订单
            if quantity > 0:
                ask_orders.append({"price": float(ask_prices[i]), "quantity": quantity})

        return ActionType.QUOTE, {"bid_orders": bid_orders, "ask_orders": ask_orders}

    def _cancel_all_orders(self, matching_engine: "MatchingEngine") -> None:
        """撤销所有挂单

        Args:
            matching_engine: 撮合引擎
        """
        for order_id in self.bid_order_ids + self.ask_order_ids:
            matching_engine.cancel_order(order_id)
        self.bid_order_ids.clear()
        self.ask_order_ids.clear()

    def _place_quote_orders(
        self,
        orders: list[dict[str, float]],
        side: OrderSide,
        order_ids: list[int],
        matching_engine: "MatchingEngine",
    ) -> list[Trade]:
        """挂限价单并记录订单ID

        Args:
            orders: 订单列表，每个订单包含 price 和 quantity
            side: 订单方向（买/卖）
            order_ids: 用于存储订单ID的列表（将被修改）
            matching_engine: 撮合引擎

        Returns:
            成交列表
        """
        all_trades: list[Trade] = []
        for order_spec in orders:
            order_id = self._generate_order_id()
            order = Order(
                order_id=order_id,
                agent_id=self.agent_id,
                side=side,
                order_type=OrderType.LIMIT,
                price=order_spec["price"],
                quantity=order_spec["quantity"],
            )
            trades = matching_engine.process_order(order)
            self._process_trades(trades)
            all_trades.extend(trades)
            order_ids.append(order_id)
        return all_trades

    def _handle_quote(
        self,
        params: dict[str, Any],
        matching_engine: "MatchingEngine",
    ) -> list[Trade]:
        """处理 QUOTE 动作

        先撤掉所有旧挂单，然后双边各挂 1-5 单（每单价格和数量由神经网络决定）。

        Args:
            params: 动作参数字典
                {"bid_orders": [{"price": float, "quantity": float}, ...],
                 "ask_orders": [{"price": float, "quantity": float}, ...]}
            matching_engine: 撮合引擎

        Returns:
            成交列表
        """
        self._cancel_all_orders(matching_engine)
        all_trades: list[Trade] = []
        all_trades.extend(
            self._place_quote_orders(
                params.get("bid_orders", []),
                OrderSide.BUY,
                self.bid_order_ids,
                matching_engine,
            )
        )
        all_trades.extend(
            self._place_quote_orders(
                params.get("ask_orders", []),
                OrderSide.SELL,
                self.ask_order_ids,
                matching_engine,
            )
        )
        return all_trades

    def _handle_clear_position(
        self,
        matching_engine: "MatchingEngine",
    ) -> list[Trade]:
        """处理做市商清仓

        先撤掉所有挂单，再根据持仓方向市价平仓。

        Args:
            matching_engine: 撮合引擎

        Returns:
            成交列表
        """
        self._cancel_all_orders(matching_engine)
        position_qty = self.account.position.quantity
        if position_qty > 0:
            return self._place_market_order(
                OrderSide.SELL, position_qty, matching_engine
            )
        elif position_qty < 0:
            return self._place_market_order(
                OrderSide.BUY, abs(position_qty), matching_engine
            )
        return []

    def execute_action(
        self,
        action: ActionType,
        params: dict[str, Any],
        matching_engine: "MatchingEngine",
    ) -> list[Trade]:
        """执行动作

        做市商特定实现：QUOTE 先撤所有旧单再双边挂单，CLEAR_POSITION 先撤单再平仓。

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

        if action == ActionType.QUOTE:
            trades = self._handle_quote(params, matching_engine)
        elif action == ActionType.CLEAR_POSITION:
            trades = self._handle_clear_position(matching_engine)

        return trades
