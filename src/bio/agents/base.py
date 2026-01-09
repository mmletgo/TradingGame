"""Agent 基类模块

本模块定义 Agent 基类，是所有 AI Agent（散户、庄家、做市商）的父类。
"""

from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np

from src.config.config import AgentConfig, AgentType

# 纯 Python 备用实现
def _py_argmax(arr: np.ndarray | list[float], start: int, end: int) -> int:
    """纯 Python argmax 实现"""
    if isinstance(arr, list):
        arr = np.array(arr, dtype=np.float64)
    return int(np.argmax(arr[start:end]))


def _py_round_price(price: float, tick_size: float) -> float:
    """纯 Python 价格取整实现"""
    return max(tick_size, round(price / tick_size) * tick_size)


def _py_clip(value: float, min_val: float, max_val: float) -> float:
    """纯 Python clip 实现"""
    return max(min_val, min(max_val, value))


# 尝试导入 Cython 加速函数，如果失败则使用纯 Python 实现
try:
    from src.bio.agents._cython.fast_decide import (
        fast_argmax as _cython_argmax,
        fast_round_price,
        fast_clip,
    )
    _HAS_CYTHON_DECIDE = True

    # 包装 Cython argmax 以自动处理 list 到 numpy 数组的转换
    def fast_argmax(arr: np.ndarray | list[float], start: int, end: int) -> int:
        """Cython 加速的 argmax，自动处理 list 输入"""
        if isinstance(arr, list):
            arr = np.array(arr, dtype=np.float64)
        return _cython_argmax(arr, start, end)
except ImportError:
    fast_argmax = _py_argmax
    fast_round_price = _py_round_price
    fast_clip = _py_clip
    _HAS_CYTHON_DECIDE = False

if TYPE_CHECKING:
    from src.market.matching.matching_engine import MatchingEngine
import neat

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

    四种类型 AI Agent（散户、高级散户、庄家、做市商）的基类，提供通用的属性和方法。

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
    config: AgentConfig
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
        self.config = config
        self.account = Account(agent_id, agent_type, config)

        # 预分配神经网络输入缓冲区（907 = 200 + 200 + 100 + 100 + 4 + 3 + 300）
        # 其中 300 = 100 tick历史价格 + 100 tick历史成交量 + 100 tick历史成交额
        self._input_buffer: np.ndarray = np.zeros(907, dtype=np.float64)

        # 预分配持仓信息缓冲区（4 个值）
        self._position_buffer: np.ndarray = np.zeros(4, dtype=np.float64)

        # 预分配挂单信息缓冲区（3 个值）
        self._pending_order_buffer: np.ndarray = np.zeros(3, dtype=np.float64)

        # 初始化强平标志
        self.is_liquidated = False

        # 初始化订单计数器
        self._order_counter = 0

    def observe(
        self, market_state: NormalizedMarketState, orderbook: OrderBook
    ) -> np.ndarray:
        """从预计算的市场状态构建神经网络输入

        Args:
            market_state: 预计算的归一化市场数据
            orderbook: 订单簿（用于查询挂单信息）

        Returns:
            神经网络输入向量（907维 ndarray）
        """
        # 直接复制到预分配数组，避免 np.concatenate 创建新数组
        self._input_buffer[:200] = market_state.bid_data
        self._input_buffer[200:400] = market_state.ask_data
        self._input_buffer[400:500] = market_state.trade_prices
        self._input_buffer[500:600] = market_state.trade_quantities
        self._input_buffer[600:604] = self._get_position_inputs(market_state.mid_price)
        self._input_buffer[604:607] = self._get_pending_order_inputs(
            market_state.mid_price, orderbook
        )
        # tick 历史数据（100 个价格 + 100 个成交量 + 100 个成交额）
        self._input_buffer[607:707] = market_state.tick_history_prices
        self._input_buffer[707:807] = market_state.tick_history_volumes
        self._input_buffer[807:907] = market_state.tick_history_amounts
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
            result[1] = (
                (self.account.position.avg_price - mid_price) / mid_price
                if mid_price > 0
                else 0.0
            )

        # 余额归一化
        initial_balance = self.account.initial_balance
        result[2] = (
            self.account.balance / initial_balance if initial_balance > 0 else 0.0
        )

        # 净值归一化
        result[3] = equity / initial_balance if initial_balance > 0 else 0.0

        return result

    def _get_pending_order_inputs(
        self, mid_price: float, orderbook: OrderBook
    ) -> np.ndarray:
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
        # 数量使用对数归一化：log10(qty + 1) / 10
        result[1] = np.log10(float(order.quantity) + 1) / 10.0
        result[2] = 1.0 if order.side == OrderSide.BUY else -1.0
        return result

    # 订单数量上限，防止 int 溢出
    MAX_ORDER_QUANTITY: int = 100_000_000

    def _calculate_order_quantity(
        self, price: float, ratio: float, is_buy: bool = True, ref_price: float = 0.0
    ) -> int:
        """计算订单数量

        根据账户净值、杠杆倍数、当前持仓和数量比例计算订单数量。
        确保下单后的总持仓市值不超过 equity * leverage。

        Args:
            price: 订单价格（用于计算最终数量）
            ratio: 数量比例（0.1 到 1.0，表示使用可用空间的比例）
            is_buy: 是否为买入方向
            ref_price: 参考价格（用于计算 equity 和仓位价值，默认为 0 则使用 price）

        Returns:
            订单数量（整数），如果净值为负或可用空间不足则返回 0
        """
        # 如果未指定参考价格，使用订单价格
        calc_price = ref_price if ref_price > 0 else price
        equity = self.account.get_equity(calc_price)

        # 净值非正时不允许下单
        if equity <= 0:
            return 0

        # 最大允许持仓市值 = 净值 * 杠杆
        max_pos_value = equity * self.account.leverage

        # 当前持仓情况
        current_pos = self.account.position.quantity  # 正数为多头，负数为空头
        current_pos_value = abs(current_pos) * calc_price

        # 计算剩余可用持仓空间
        if is_buy:
            if current_pos >= 0:
                # 当前是多头或空仓，买入是同向加仓
                available_pos_value = max(0, max_pos_value - current_pos_value)
            else:
                # 当前是空头，买入是反向平仓+可能开多仓
                # 剩余可用 = 可平仓市值 + 最大可开多仓市值
                available_pos_value = current_pos_value + max_pos_value
        else:
            if current_pos <= 0:
                # 当前是空头或空仓，卖出是同向加仓
                available_pos_value = max(0, max_pos_value - current_pos_value)
            else:
                # 当前是多头，卖出是反向平仓+可能开空仓
                # 剩余可用 = 可平仓市值 + 最大可开空仓市值
                available_pos_value = current_pos_value + max_pos_value

        # 限制比例在合理范围
        ratio = min(1.0, ratio)
        quantity = (available_pos_value * ratio) / price if price > 0 else 0.0

        # 确保数量为整数且至少为1（最小交易单位），同时限制最大值防止溢出
        if quantity < 1:
            return 0
        quantity = min(self.MAX_ORDER_QUANTITY, int(quantity))

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

        先撤掉挂单，再根据持仓方向市价平仓。
        做市商重写此方法以处理多个挂单。

        Args:
            matching_engine: 撮合引擎

        Returns:
            成交列表
        """
        # 先撤掉挂单
        if self.account.pending_order_id is not None:
            matching_engine.cancel_order(self.account.pending_order_id)
            self.account.pending_order_id = None

        # 再根据持仓方向市价平仓
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

    def update_brain(
        self, genome: neat.DefaultGenome, config: neat.Config
    ) -> None:
        """原地更新 brain，复用 Agent 对象

        Args:
            genome: 新的 NEAT 基因组
            config: NEAT 配置
        """
        self.brain.update_from_genome(genome, config)
