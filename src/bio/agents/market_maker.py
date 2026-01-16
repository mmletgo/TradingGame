"""做市商 Agent 模块

本模块定义做市商 Agent 类，继承自 Agent 基类。
"""

from typing import TYPE_CHECKING, Any
import logging

import numpy as np

# 调试日志开关
_DEBUG_EMPTY_ORDERS = False  # 关闭调试日志
_debug_logger = logging.getLogger("market_maker_debug")
_debug_logger.setLevel(logging.WARNING)

from src.bio.agents.base import Agent, ActionType
from src.bio.brain.brain import Brain
from src.config.config import AgentConfig, AgentType
from src.market.market_state import NormalizedMarketState
from src.market.orderbook.order import Order, OrderSide, OrderType
from src.market.orderbook.orderbook import OrderBook
from src.market.matching.trade import Trade

if TYPE_CHECKING:
    from src.market.matching.matching_engine import MatchingEngine

# 尝试导入 Cython 加速的 observe 函数
try:
    from src.bio.agents._cython.fast_observe import (
        fast_observe_market_maker,
        get_position_inputs as cython_get_position_inputs,
    )
    _HAS_CYTHON_OBSERVE = True
except ImportError:
    _HAS_CYTHON_OBSERVE = False


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

    神经网络输入维度: 964 = 604 + 60 挂单信息 + 300 tick 历史数据
    神经网络输出维度: 41 = 买单价格(10) + 买单数量(10) + 卖单价格(10) + 卖单数量(10) + 总下单比例基准(1)
    """

    agent_id: int
    brain: Brain
    bid_order_ids: list[int]
    ask_order_ids: list[int]

    def __init__(self, agent_id: int, brain: Brain, config: AgentConfig) -> None:
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

        # 做市商输入更大（964 = 604 + 60 挂单信息 + 300 tick 历史数据）
        # tick 历史数据：100 个价格 + 100 个成交量 + 100 个成交额
        self._input_buffer = np.zeros(964, dtype=np.float64)

    def get_action_space(self) -> list[ActionType]:
        """获取做市商可用动作

        做市商默认每 tick 双边挂单，无需动作选择。

        Returns:
            空列表（做市商不使用动作选择）
        """
        return []

    def observe(
        self, market_state: NormalizedMarketState, orderbook: OrderBook
    ) -> np.ndarray:
        """从预计算的市场状态构建神经网络输入

        做市商覆盖基类方法，使用更大的输入缓冲区（964 = 604 + 60 挂单信息 + 300 tick 历史数据）。

        Args:
            market_state: 预计算的归一化市场数据
            orderbook: 订单簿（用于查询挂单信息）

        Returns:
            神经网络输入向量（964 维 ndarray）
        """
        if _HAS_CYTHON_OBSERVE:
            # 使用 Cython 加速版本
            mid_price = market_state.mid_price
            equity = self.account.get_equity(mid_price)
            position_inputs = cython_get_position_inputs(
                equity,
                self.account.leverage,
                self.account.position.quantity,
                self.account.position.avg_price,
                self.account.balance,
                self.account.initial_balance,
                mid_price,
            )

            # 获取挂单信息（做市商特有：60个值）
            pending_order_inputs = self._get_pending_order_inputs(
                mid_price, orderbook
            )

            # 调用 Cython 函数填充缓冲区
            fast_observe_market_maker(
                self._input_buffer,
                market_state.bid_data,
                market_state.ask_data,
                market_state.trade_prices,
                market_state.trade_quantities,
                market_state.tick_history_prices,
                market_state.tick_history_volumes,
                market_state.tick_history_amounts,
                position_inputs[0],
                position_inputs[1],
                position_inputs[2],
                position_inputs[3],
                pending_order_inputs,
            )
            return self._input_buffer
        else:
            # 纯 Python 实现：直接复制到预分配数组
            self._input_buffer[:200] = market_state.bid_data
            self._input_buffer[200:400] = market_state.ask_data
            self._input_buffer[400:500] = market_state.trade_prices
            self._input_buffer[500:600] = market_state.trade_quantities
            self._input_buffer[600:604] = self._get_position_inputs(market_state.mid_price)
            self._input_buffer[604:664] = self._get_pending_order_inputs(
                market_state.mid_price, orderbook
            )
            # tick 历史数据（100 个价格 + 100 个成交量 + 100 个成交额）
            self._input_buffer[664:764] = market_state.tick_history_prices
            self._input_buffer[764:864] = market_state.tick_history_volumes
            self._input_buffer[864:964] = market_state.tick_history_amounts
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
        for i in range(10):
            if i < len(order_ids):
                order = orderbook.order_map.get(order_ids[i])
                if order is not None:
                    price_norm = (
                        (order.price - mid_price) / mid_price if mid_price > 0 else 0.0
                    )
                    idx = offset + i * 3
                    inputs[idx] = price_norm
                    # 数量使用对数归一化：log10(qty + 1) / 10
                    inputs[idx + 1] = np.log10(float(order.quantity) + 1) / 10.0
                    inputs[idx + 2] = 1.0

    def _get_pending_order_inputs(
        self, mid_price: float, orderbook: OrderBook
    ) -> np.ndarray:
        """获取做市商挂单信息（60 个值）

        买单 10 个位置 x 3 = 30
        卖单 10 个位置 x 3 = 30
        每个位置：[价格归一化, 数量, 有效标志(1.0/0.0)]

        Args:
            mid_price: 中间价（用于价格归一化）
            orderbook: 订单簿（用于查询订单详情）

        Returns:
            60 个浮点数的 NumPy 数组
        """
        inputs = np.zeros(60, dtype=np.float64)
        self._fill_order_inputs(inputs, self.bid_order_ids, 0, mid_price, orderbook)
        self._fill_order_inputs(inputs, self.ask_order_ids, 30, mid_price, orderbook)
        return inputs

    def _calculate_skew_factor(self, mid_price: float) -> float:
        """计算仓位倾斜因子

        根据当前仓位计算倾斜因子，范围 [-1, 1]。
        - 多头仓位 -> 负值（减少买单权重，增加卖单权重）
        - 空头仓位 -> 正值（增加买单权重，减少卖单权重）

        Args:
            mid_price: 中间价

        Returns:
            倾斜因子，范围 [-1, 1]
        """
        equity = self.account.get_equity(mid_price)
        if equity <= 0:
            return 0.0

        position_qty = self.account.position.quantity
        if position_qty == 0:
            return 0.0

        # 计算仓位比例（0 到 1）
        position_value = abs(position_qty) * mid_price
        max_position_value = equity * self.account.leverage
        pos_ratio = (
            min(1.0, position_value / max_position_value)
            if max_position_value > 0
            else 0.0
        )

        # 多头为负（倾向卖出），空头为正（倾向买入）
        if position_qty > 0:
            return -pos_ratio
        else:
            return pos_ratio

    def _apply_position_skew(
        self,
        bid_raw_ratios: np.ndarray,
        ask_raw_ratios: np.ndarray,
        skew_factor: float,
        min_side_weight: float = 0.03,
    ) -> tuple[np.ndarray, np.ndarray]:
        """应用仓位倾斜到买卖权重

        Args:
            bid_raw_ratios: 买单原始权重
            ask_raw_ratios: 卖单原始权重
            skew_factor: 倾斜因子 [-1, 1]
            min_side_weight: 单边最小权重比例

        Returns:
            (bid_ratios, ask_ratios): 调整后的买卖权重，总和为 1.0
        """
        # 计算倾斜乘数
        bid_multiplier = 1.0 + skew_factor  # 范围 [0, 2]
        ask_multiplier = 1.0 - skew_factor  # 范围 [0, 2]

        # 应用乘数
        bid_adjusted = bid_raw_ratios * bid_multiplier
        ask_adjusted = ask_raw_ratios * ask_multiplier

        # 计算调整后的总权重
        total_bid = bid_adjusted.sum()
        total_ask = ask_adjusted.sum()
        total = total_bid + total_ask

        if total <= 0:
            return np.full(10, 0.1), np.full(10, 0.1)

        # 计算双边比例
        bid_side_ratio = total_bid / total
        ask_side_ratio = total_ask / total

        # 确保最小权重
        if bid_side_ratio < min_side_weight:
            target_bid_total = min_side_weight
            target_ask_total = 1.0 - min_side_weight
        elif ask_side_ratio < min_side_weight:
            target_ask_total = min_side_weight
            target_bid_total = 1.0 - min_side_weight
        else:
            target_bid_total = bid_side_ratio
            target_ask_total = ask_side_ratio

        # 内部归一化
        if total_bid > 0:
            bid_ratios = bid_adjusted / total_bid * target_bid_total
        else:
            bid_ratios = np.full(10, target_bid_total / 10)

        if total_ask > 0:
            ask_ratios = ask_adjusted / total_ask * target_ask_total
        else:
            ask_ratios = np.full(10, target_ask_total / 10)

        return bid_ratios, ask_ratios

    def decide(
        self, market_state: NormalizedMarketState, orderbook: OrderBook
    ) -> tuple[ActionType, dict[str, Any]]:
        """决策下一步动作

        做市商默认每 tick 双边挂单，神经网络直接输出价格和数量参数。
        神经网络输出结构（共 41 个值）：
        - 输出[0-9]: 买单1-10的价格偏移（-1到1，相对于mid_price）
        - 输出[10-19]: 买单1-10的数量权重（-1到1，映射到0-1，20单归一化后总和为1.0）
        - 输出[20-29]: 卖单1-10的价格偏移（-1到1，相对于mid_price）
        - 输出[30-39]: 卖单1-10的数量权重（-1到1，映射到0-1，20单归一化后总和为1.0）
        - 输出[40]: 总下单比例基准（-1到1，映射到0-1，控制使用多少可用资金下单）

        Args:
            market_state: 预计算的归一化市场数据
            orderbook: 订单簿对象

        Returns:
            (动作类型, 动作参数字典)
            - HOLD: {"bid_orders": [{"price": float, "quantity": float}, ...],
                     "ask_orders": [{"price": float, "quantity": float}, ...]}
        """
        # 如果已被强平，返回空订单列表
        if self.is_liquidated:
            return ActionType.HOLD, {"bid_orders": [], "ask_orders": []}

        # 1. 观察市场，获取神经网络输入（复用基类方法）
        inputs = self.observe(market_state, orderbook)

        # 2. 神经网络前向传播
        outputs = self.brain.forward(inputs)

        # 3. 验证输出维度（需要 41 个值）
        if len(outputs) < 41:
            raise ValueError(f"神经网络输出维度不足，期望 41，实际 {len(outputs)}")

        # 4. 获取参考价格并计算仓位信息
        mid_price = market_state.mid_price
        if mid_price == 0:
            mid_price = 100.0

        tick_size = market_state.tick_size if market_state.tick_size > 0 else 0.01

        # 计算仓位倾斜因子
        skew_factor = self._calculate_skew_factor(mid_price)
        position_qty = self.account.position.quantity

        # 向量化收集数量比例
        outputs_arr = np.array(outputs)
        bid_raw_ratios = np.maximum(0.0, (np.clip(outputs_arr[10:20], -1, 1) + 1) / 2)
        ask_raw_ratios = np.maximum(0.0, (np.clip(outputs_arr[30:40], -1, 1) + 1) / 2)

        # 计算总和并归一化（确保 20 个订单的总比例 = 1.0）
        total_raw_ratio = bid_raw_ratios.sum() + ask_raw_ratios.sum()
        if total_raw_ratio > 0:
            bid_ratios = bid_raw_ratios / total_raw_ratio
            ask_ratios = ask_raw_ratios / total_raw_ratio
        else:
            bid_ratios = np.zeros(10)
            ask_ratios = np.zeros(10)

        # 应用仓位倾斜
        bid_ratios, ask_ratios = self._apply_position_skew(
            bid_ratios, ask_ratios, skew_factor
        )

        # 解析总下单比例基准
        # 输出[40]: -1 到 1，映射到 0.01 到 1
        total_ratio_base = (
            0.01 + (np.clip(outputs_arr[40], -1, 1) + 1) / 2 * 0.99
        )  # [0.01, 1]

        # 应用总下单比例基准到权重
        bid_ratios = bid_ratios * total_ratio_base
        ask_ratios = ask_ratios * total_ratio_base

        # 价格偏移完全由神经网络决定，映射到 [1, 100] ticks
        max_offset_ticks = 100.0
        min_offset_ticks = 1.0

        bid_price_offsets = np.clip(outputs_arr[0:10], -1, 1)
        # [-1, 1] -> [1, 100]
        bid_price_ticks = min_offset_ticks + (bid_price_offsets + 1) / 2 * (
            max_offset_ticks - min_offset_ticks
        )
        # 舍入到 tick_size 的整数倍，避免浮点数精度问题
        # 确保价格至少为一个 tick_size，防止出现负价格或零价格
        bid_prices = np.maximum(
            tick_size,
            np.round((mid_price - bid_price_ticks * tick_size) / tick_size) * tick_size,
        )

        bid_orders: list[dict[str, float]] = []
        for i in range(10):
            quantity = self._calculate_order_quantity(
                float(bid_prices[i]),
                float(bid_ratios[i]),
                is_buy=True,
                ref_price=mid_price,
            )
            if quantity > 0:
                bid_orders.append({"price": float(bid_prices[i]), "quantity": quantity})

        # 卖单价格偏移完全由神经网络决定
        ask_price_offsets = np.clip(outputs_arr[20:30], -1, 1)
        # [-1, 1] -> [1, 100]
        ask_price_ticks = min_offset_ticks + (ask_price_offsets + 1) / 2 * (
            max_offset_ticks - min_offset_ticks
        )
        # 舍入到 tick_size 的整数倍，避免浮点数精度问题
        ask_prices = (
            np.round((mid_price + ask_price_ticks * tick_size) / tick_size) * tick_size
        )

        ask_orders: list[dict[str, float]] = []
        for i in range(10):
            quantity = self._calculate_order_quantity(
                float(ask_prices[i]),
                float(ask_ratios[i]),
                is_buy=False,
                ref_price=mid_price,
            )
            if quantity > 0:
                ask_orders.append({"price": float(ask_prices[i]), "quantity": quantity})

        # 调试日志：检测空订单列表或无仓位但订单为空的情况
        should_log = len(bid_orders) == 0 or len(ask_orders) == 0
        if _DEBUG_EMPTY_ORDERS and should_log:
            equity = self.account.get_equity(mid_price)
            max_pos = equity * self.account.leverage if equity > 0 else 0
            current_pos_value = abs(position_qty) * mid_price

            # 计算买卖方向的可用空间
            if position_qty >= 0:  # 多头或空仓
                buy_available = max(0, max_pos - current_pos_value)
                sell_available = current_pos_value + max_pos
            else:  # 空头
                buy_available = current_pos_value + max_pos
                sell_available = max(0, max_pos - current_pos_value)

            # 额外输出 raw ratios 信息，帮助排查
            _debug_logger.warning(
                f"MM {self.agent_id} 空订单: "
                f"bid_orders={len(bid_orders)}, ask_orders={len(ask_orders)}, "
                f"pos={position_qty}, equity={equity:.0f}, "
                f"pos_value={current_pos_value:.0f}, max_pos={max_pos:.0f}, "
                f"buy_avail={buy_available:.0f}, sell_avail={sell_available:.0f}, "
                f"skew={skew_factor:.2f}, "
                f"bid_ratios=[{', '.join(f'{r:.4f}' for r in bid_ratios)}], "
                f"ask_ratios=[{', '.join(f'{r:.4f}' for r in ask_ratios)}], "
                f"bid_prices=[{', '.join(f'{p:.1f}' for p in bid_prices)}], "
                f"ask_prices=[{', '.join(f'{p:.1f}' for p in ask_prices)}]"
            )

        return ActionType.HOLD, {"bid_orders": bid_orders, "ask_orders": ask_orders}

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
                quantity=int(order_spec["quantity"]),
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

    def execute_action(
        self,
        action: ActionType,
        params: dict[str, Any],
        matching_engine: "MatchingEngine",
    ) -> list[Trade]:
        """执行动作

        做市商默认每 tick 双边挂单，先撤所有旧单再挂新单。

        Args:
            action: 动作类型（忽略此参数，做市商始终执行双边挂单）
            params: 动作参数字典
            matching_engine: 撮合引擎

        Returns:
            成交列表
        """
        if self.is_liquidated:
            return []

        return self._handle_quote(params, matching_engine)
