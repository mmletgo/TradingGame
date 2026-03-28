"""回放引擎核心模块

在真实市场数据流上模拟单个 agent 的交易。
每个 step = 一个订单簿快照间隔，处理 step 间的所有成交事件（被动成交检查）。
支持散户和做市商两种 agent 类型。

典型使用流程:
    engine = ReplayEngine(config)
    engine.load_data(ob_snapshots, trades)
    state = engine.reset(start_idx=0)
    while True:
        action = agent.forward(state)
        result = engine.step(action)
        if result.done:
            break
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass, field
from math import log
from typing import Any

import numpy as np

from src.config.config import AgentConfig, AgentType
from src.market.account.account import Account
from src.market.matching.trade import Trade
from src.market.market_state import NormalizedMarketState
from src.replay.config import ReplayConfig
from src.replay.data_loader import MarketTrade, OrderbookSnapshot
from src.replay.fill_model import FillModel, PendingOrder
from src.replay.market_state_builder import MarketStateBuilder


# ---------------------------------------------------------------------------
# 订单数量上限，与 Agent 基类保持一致
# ---------------------------------------------------------------------------
MAX_ORDER_QUANTITY: int = 100_000_000


@dataclass
class StepResult:
    """单步结果

    Attributes:
        market_state: 当前 tick 的归一化市场状态
        reward: 本步 reward
        done: 是否结束
        info: 附加信息字典
    """

    market_state: NormalizedMarketState
    reward: float
    done: bool
    info: dict[str, Any] = field(default_factory=dict)


class ReplayEngine:
    """实盘回放引擎

    在真实市场数据流上模拟单个 agent 的交易:
    - 每个 step = 一个订单簿快照
    - 处理 step 间的所有成交事件 (被动成交检查)
    - 支持散户和做市商两种 agent 类型

    数据流:
        load_data() -> reset() -> step() -> step() -> ... -> done
    """

    def __init__(self, config: ReplayConfig) -> None:
        self._config: ReplayConfig = config
        self._fill_model: FillModel = FillModel(config)
        self._state_builder: MarketStateBuilder = MarketStateBuilder(config)

        # 数据 (由 load_data 设置)
        self._ob_snapshots: list[OrderbookSnapshot] = []
        self._trades: list[MarketTrade] = []
        # 用于二分查找的成交时间戳数组
        self._trade_timestamps: list[int] = []

        # 运行时状态
        self._account: Account | None = None
        self._pending_orders: list[PendingOrder] = []
        self._order_counter: int = 0
        self._current_ob_idx: int = 0
        self._current_trade_idx: int = 0
        self._step_count: int = 0
        self._prev_equity: float = 0.0
        self._current_ob: OrderbookSnapshot | None = None
        self._trade_id_counter: int = 0

    # ------------------------------------------------------------------
    # 数据管理
    # ------------------------------------------------------------------

    def load_data(
        self,
        ob_snapshots: list[OrderbookSnapshot],
        trades: list[MarketTrade],
    ) -> None:
        """设置回放数据

        Args:
            ob_snapshots: 订单簿快照列表 (按 timestamp_ms 升序)
            trades: 逐笔成交列表 (按 timestamp_ms 升序)
        """
        self._ob_snapshots = ob_snapshots
        self._trades = trades
        # 预提取时间戳用于二分查找
        self._trade_timestamps = [t.timestamp_ms for t in trades]

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    def reset(self, start_idx: int = 0) -> NormalizedMarketState:
        """重置引擎状态

        Args:
            start_idx: 订单簿快照流中的起始索引

        Returns:
            初始 NormalizedMarketState
        """
        # 创建 Account
        agent_type: AgentType = (
            AgentType.RETAIL_PRO
            if self._config.agent_type == "RETAIL_PRO"
            else AgentType.MARKET_MAKER
        )
        agent_config: AgentConfig = AgentConfig(
            count=1,
            initial_balance=self._config.initial_balance,
            leverage=self._config.leverage,
            maintenance_margin_rate=self._config.maintenance_margin_rate,
            maker_fee_rate=self._config.maker_fee_rate,
            taker_fee_rate=self._config.taker_fee_rate,
        )
        self._account = Account(
            agent_id=0, agent_type=agent_type, config=agent_config
        )

        # 重置状态
        self._pending_orders = []
        self._order_counter = 0
        self._trade_id_counter = 0
        self._current_ob_idx = start_idx
        self._step_count = 0
        self._state_builder.reset()

        # 找到起始 OB 对应的 trade 索引
        if start_idx < len(self._ob_snapshots):
            start_ts: int = self._ob_snapshots[start_idx].timestamp_ms
            self._current_trade_idx = self._find_trade_idx(start_ts)
        else:
            self._current_trade_idx = len(self._trades)

        # 构建初始状态
        self._current_ob = self._ob_snapshots[self._current_ob_idx]
        self._state_builder.update_tick(self._current_ob, 0.0, 0.0)
        state: NormalizedMarketState = self._state_builder.build(self._current_ob)
        self._prev_equity = self._config.initial_balance
        self._current_ob_idx += 1

        return state

    def step(self, action: np.ndarray) -> StepResult:
        """执行一步

        流程:
        1. 解析 agent 动作并执行 (挂单/市价单)
        2. 推进到下一个 OB 快照
        3. 处理期间的成交事件 (被动成交)
        4. 构建新状态
        5. 计算 reward
        6. 检查终止条件

        Args:
            action: 神经网络原始输出
                - 散户: shape (3,) -- [动作选择, 价格偏移, 数量比例]
                - 做市商: shape (43,) -- [bid_offsets(10), bid_weights(10),
                  ask_offsets(10), ask_weights(10), total_ratio(1),
                  gamma_adj(1), spread_adj(1)]

        Returns:
            StepResult 包含新状态、reward、终止标志和附加信息
        """
        assert self._account is not None, "必须先调用 reset()"
        assert self._current_ob is not None, "必须先调用 reset()"

        done: bool = False
        mid_price: float = self._current_ob.mid_price

        # 1. 解析并执行 agent 动作
        self._execute_action(action, self._current_ob)

        # 2. 推进到下一个 OB 快照
        if self._current_ob_idx >= len(self._ob_snapshots):
            done = True
            # 最后一步，用当前 OB 构建状态
            state: NormalizedMarketState = self._state_builder.build(self._current_ob)
        else:
            next_ob: OrderbookSnapshot = self._ob_snapshots[self._current_ob_idx]

            # 3. 处理两个 OB 之间的所有成交
            tick_volume: float = 0.0
            tick_amount: float = 0.0
            trades_between: list[MarketTrade] = []

            while (
                self._current_trade_idx < len(self._trades)
                and self._trades[self._current_trade_idx].timestamp_ms
                <= next_ob.timestamp_ms
            ):
                trade: MarketTrade = self._trades[self._current_trade_idx]

                # 检查被动成交
                fills: list[tuple[PendingOrder, int]] = (
                    self._fill_model.check_passive_fills(
                        self._pending_orders, trade
                    )
                )
                for order, fill_qty in fills:
                    self._process_fill(order, fill_qty, trade.price)
                    order.quantity -= fill_qty

                # 移除完全成交的挂单
                self._pending_orders = [
                    o for o in self._pending_orders if o.quantity > 0
                ]

                trades_between.append(trade)

                # 累计 tick 成交量/额 (带方向)
                sign: float = 1.0 if trade.side == "buy" else -1.0
                tick_volume += sign * trade.amount
                tick_amount += sign * trade.amount * trade.price

                self._current_trade_idx += 1

            # 4. 更新状态
            self._current_ob = next_ob
            mid_price = next_ob.mid_price
            self._state_builder.add_trades(trades_between, mid_price)
            self._state_builder.update_tick(next_ob, tick_volume, tick_amount)
            state = self._state_builder.build(next_ob)
            self._current_ob_idx += 1

        # 5. 计算 reward
        curr_equity: float = self._account.get_equity(mid_price)
        initial: float = self._config.initial_balance

        # reward = delta_equity_rate - position_cost
        delta_equity: float = curr_equity - self._prev_equity
        reward: float = delta_equity / initial if initial > 0 else 0.0

        # 持仓成本惩罚: lambda * |position_value| / initial
        pos_value: float = abs(self._account.position.quantity) * mid_price
        reward -= self._config.position_cost_weight * pos_value / initial

        self._prev_equity = curr_equity
        self._step_count += 1

        # 6. 检查终止条件
        if self._config.episode_length > 0 and self._step_count >= self._config.episode_length:
            done = True
        if self._account.check_liquidation(mid_price):
            done = True

        # 7. Episode 结束时平仓计算最终 PnL
        if done and self._account.position.quantity != 0:
            self._force_close(mid_price)

        info: dict[str, Any] = {
            "equity": curr_equity,
            "balance": self._account.balance,
            "position": self._account.position.quantity,
            "step": self._step_count,
            "mid_price": mid_price,
        }

        return StepResult(
            market_state=state, reward=reward, done=done, info=info
        )

    # ------------------------------------------------------------------
    # 动作执行
    # ------------------------------------------------------------------

    def _execute_action(
        self, action: np.ndarray, ob: OrderbookSnapshot
    ) -> None:
        """解析并执行 agent 动作

        Args:
            action: 神经网络原始输出
            ob: 当前订单簿快照
        """
        if self._config.agent_type == "RETAIL_PRO":
            self._execute_retail_action(action, ob)
        else:
            self._execute_mm_action(action, ob)

    def _execute_retail_action(
        self, action: np.ndarray, ob: OrderbookSnapshot
    ) -> None:
        """解析散户动作 (3 输出)

        复用 RetailProAgent.decide() 的逻辑:
        - action[0]: 动作选择 [-1,1] -> 等宽分 6 bin
        - action[1]: 价格偏移 [-1,1] -> +/-100 ticks
        - action[2]: 数量比例 [-1,1] -> [0, 1.0]

        动作类型:
        0=HOLD, 1=PLACE_BID, 2=PLACE_ASK, 3=CANCEL,
        4=MARKET_BUY, 5=MARKET_SELL

        Args:
            action: shape (3,) 的动作向量
            ob: 当前订单簿快照
        """
        mid_price: float = ob.mid_price
        if mid_price <= 0:
            return
        tick_size: float = self._config.tick_size

        # 解析动作类型 (与 RetailProAgent.decide 完全一致)
        action_value: float = float(np.clip(action[0], -1.0, 1.0))
        action_idx: int = min(5, int((action_value + 1.0) * 3.0))

        # 解析参数
        price_offset_norm: float = float(np.clip(action[1], -1.0, 1.0))
        quantity_ratio_norm: float = float(np.clip(action[2], -1.0, 1.0))
        quantity_ratio: float = (quantity_ratio_norm + 1.0) * 0.5  # -> [0, 1]

        if action_idx == 0:
            # HOLD: 不操作
            pass

        elif action_idx == 1:
            # PLACE_BID: 挂买单
            price: float = _round_price(
                mid_price + price_offset_norm * 100 * tick_size, tick_size
            )
            qty: int = self._calc_quantity(mid_price, quantity_ratio, is_buy=True)
            if qty > 0:
                self._cancel_all_pending()
                self._add_pending(1, price, qty, ob.timestamp_ms)

        elif action_idx == 2:
            # PLACE_ASK: 挂卖单
            price = _round_price(
                mid_price + price_offset_norm * 100 * tick_size, tick_size
            )
            qty = self._calc_quantity(mid_price, quantity_ratio, is_buy=False)
            if qty > 0:
                self._cancel_all_pending()
                self._add_pending(-1, price, qty, ob.timestamp_ms)

        elif action_idx == 3:
            # CANCEL: 撤单
            self._cancel_all_pending()

        elif action_idx == 4:
            # MARKET_BUY: 市价买入
            qty = self._calc_quantity(mid_price, quantity_ratio, is_buy=True)
            if qty > 0:
                fill_price, fill_qty = self._fill_model.check_active_fill(
                    1, qty, ob
                )
                if fill_qty > 0:
                    self._process_market_fill(1, fill_price, fill_qty)

        elif action_idx == 5:
            # MARKET_SELL: 市价卖出
            qty = self._calc_quantity(mid_price, quantity_ratio, is_buy=False)
            if qty > 0:
                fill_price, fill_qty = self._fill_model.check_active_fill(
                    -1, qty, ob
                )
                if fill_qty > 0:
                    self._process_market_fill(-1, fill_price, fill_qty)

    def _execute_mm_action(
        self, action: np.ndarray, ob: OrderbookSnapshot
    ) -> None:
        """解析做市商动作 (43 输出)

        复用 MarketMakerAgent.decide() 的逻辑:
        - [0:10]  买单价格偏移
        - [10:20] 买单数量权重
        - [20:30] 卖单价格偏移
        - [30:40] 卖单数量权重
        - [40]    总下单比例
        - [41]    gamma_adjustment (此处简化忽略)
        - [42]    spread_adjustment (此处简化忽略)

        做市商每步: 撤所有旧单 -> 双边各挂 1-10 单。
        简化处理: 不含 AS 模型计算，使用 mid_price 作为报价中心。

        Args:
            action: shape (43,) 的动作向量
            ob: 当前订单簿快照
        """
        # 撤所有旧单
        self._cancel_all_pending()

        mid_price: float = ob.mid_price
        if mid_price <= 0:
            return
        tick_size: float = self._config.tick_size

        outputs: np.ndarray = np.clip(action, -1.0, 1.0)

        # 数量权重解析 (与 MarketMakerAgent.decide 完全一致)
        bid_raw_ratios: np.ndarray = np.maximum(
            0.0, (np.clip(outputs[10:20], -1, 1) + 1) / 2
        )
        ask_raw_ratios: np.ndarray = np.maximum(
            0.0, (np.clip(outputs[30:40], -1, 1) + 1) / 2
        )

        # 归一化 (确保 20 个订单的总比例 = 1.0)
        total_raw_ratio: float = float(bid_raw_ratios.sum() + ask_raw_ratios.sum())
        if total_raw_ratio > 0:
            bid_ratios: np.ndarray = bid_raw_ratios / total_raw_ratio
            ask_ratios: np.ndarray = ask_raw_ratios / total_raw_ratio
        else:
            bid_ratios = np.zeros(10)
            ask_ratios = np.zeros(10)

        # 总下单比例基准: [-1, 1] -> [0.01, 1.0]
        total_ratio_base: float = float(
            0.01 + (np.clip(outputs[40], -1, 1) + 1) / 2 * 0.99
        )
        bid_ratios = bid_ratios * total_ratio_base
        ask_ratios = ask_ratios * total_ratio_base

        # 价格偏移: 简化使用 min_offset_ticks=1, max_offset_ticks=100
        min_offset_ticks: float = 1.0
        max_offset_ticks: float = min_offset_ticks + 99.0

        # 买单
        bid_price_offsets: np.ndarray = np.clip(outputs[0:10], -1, 1)
        for i in range(10):
            offset_ticks: float = min_offset_ticks + (
                float(bid_price_offsets[i]) + 1.0
            ) / 2.0 * (max_offset_ticks - min_offset_ticks)
            price: float = _round_price(
                mid_price - offset_ticks * tick_size, tick_size
            )
            qty: int = self._calc_quantity(
                mid_price, float(bid_ratios[i]), is_buy=True
            )
            if qty > 0 and price > 0:
                self._add_pending(1, price, qty, ob.timestamp_ms)

        # 卖单
        ask_price_offsets: np.ndarray = np.clip(outputs[20:30], -1, 1)
        for i in range(10):
            offset_ticks = min_offset_ticks + (
                float(ask_price_offsets[i]) + 1.0
            ) / 2.0 * (max_offset_ticks - min_offset_ticks)
            price = _round_price(
                mid_price + offset_ticks * tick_size, tick_size
            )
            qty = self._calc_quantity(
                mid_price, float(ask_ratios[i]), is_buy=False
            )
            if qty > 0:
                self._add_pending(-1, price, qty, ob.timestamp_ms)

    # ------------------------------------------------------------------
    # 订单量计算 (复用 Agent._calculate_order_quantity 逻辑)
    # ------------------------------------------------------------------

    def _calc_quantity(
        self, ref_price: float, ratio: float, is_buy: bool
    ) -> int:
        """计算订单量

        与 Agent._calculate_order_quantity 完全一致的逻辑:
        根据账户净值、杠杆、当前持仓和比例计算订单数量。
        确保下单后的总持仓市值不超过 equity * leverage。

        Args:
            ref_price: 参考价格 (mid_price)
            ratio: 数量比例 (0.0 到 1.0)
            is_buy: 是否为买入方向

        Returns:
            订单数量 (整数)，净值为负或可用空间不足则返回 0
        """
        assert self._account is not None

        if ref_price <= 0:
            return 0

        equity: float = self._account.get_equity(ref_price)
        if equity <= 0:
            return 0

        # 最大允许持仓市值 = 净值 * 杠杆
        max_pos_value: float = equity * self._account.leverage
        current_pos: int = self._account.position.quantity
        current_pos_value: float = abs(current_pos) * ref_price

        # 计算剩余可用持仓空间
        available_pos_value: float
        if is_buy:
            if current_pos >= 0:
                # 多头或空仓，买入是同向加仓
                available_pos_value = max(0.0, max_pos_value - current_pos_value)
            else:
                # 空头，买入是反向平仓 + 可能开多仓
                available_pos_value = current_pos_value + max_pos_value
        else:
            if current_pos <= 0:
                # 空头或空仓，卖出是同向加仓
                available_pos_value = max(0.0, max_pos_value - current_pos_value)
            else:
                # 多头，卖出是反向平仓 + 可能开空仓
                available_pos_value = current_pos_value + max_pos_value

        # 限制比例在 [0, 1]
        ratio = max(0.0, min(1.0, ratio))
        quantity_f: float = (available_pos_value * ratio) / ref_price

        if quantity_f < 1.0:
            return 0
        return min(MAX_ORDER_QUANTITY, int(quantity_f))

    # ------------------------------------------------------------------
    # 成交处理
    # ------------------------------------------------------------------

    def _process_fill(
        self, order: PendingOrder, fill_qty: int, fill_price: float
    ) -> None:
        """处理被动成交 (maker)

        构造 Trade 对象，调用 account.on_trade()。

        Args:
            order: 被触发的挂单
            fill_qty: 成交数量
            fill_price: 成交价格
        """
        assert self._account is not None

        self._trade_id_counter += 1
        is_buy: bool = order.side == 1

        # 计算手续费 (agent 作为 maker)
        trade_amount: float = fill_price * fill_qty
        fee: float = trade_amount * self._config.maker_fee_rate

        # 构造 Trade (agent 是 maker，所以 is_buyer_taker 与 agent 方向相反)
        trade: Trade = Trade(
            trade_id=self._trade_id_counter,
            price=fill_price,
            quantity=fill_qty,
            buyer_id=0 if is_buy else -1,
            seller_id=-1 if is_buy else 0,
            buyer_fee=fee if is_buy else 0.0,
            seller_fee=fee if not is_buy else 0.0,
            is_buyer_taker=not is_buy,  # agent 是 maker，对手是 taker
            timestamp=0.0,
        )
        self._account.on_trade(trade, is_buyer=is_buy)

    def _process_market_fill(
        self, side: int, fill_price: float, fill_qty: int
    ) -> None:
        """处理主动成交 (taker)

        构造 Trade 对象，调用 account.on_trade()。

        Args:
            side: 1=BUY, -1=SELL
            fill_price: 成交价格
            fill_qty: 成交数量
        """
        assert self._account is not None

        self._trade_id_counter += 1
        is_buy: bool = side == 1

        # 计算手续费 (agent 作为 taker)
        trade_amount: float = fill_price * fill_qty
        fee: float = trade_amount * self._config.taker_fee_rate

        # 构造 Trade (agent 是 taker)
        trade: Trade = Trade(
            trade_id=self._trade_id_counter,
            price=fill_price,
            quantity=fill_qty,
            buyer_id=0 if is_buy else -1,
            seller_id=-1 if is_buy else 0,
            buyer_fee=fee if is_buy else 0.0,
            seller_fee=fee if not is_buy else 0.0,
            is_buyer_taker=is_buy,  # agent 是 taker
            timestamp=0.0,
        )
        self._account.on_trade(trade, is_buyer=is_buy)

    # ------------------------------------------------------------------
    # 挂单管理
    # ------------------------------------------------------------------

    def _add_pending(
        self, side: int, price: float, qty: int, ts: int
    ) -> None:
        """添加挂单

        Args:
            side: 1=BUY, -1=SELL
            price: 挂单价格
            qty: 挂单数量
            ts: 时间戳
        """
        self._order_counter += 1
        self._pending_orders.append(
            PendingOrder(
                order_id=self._order_counter,
                side=side,
                price=price,
                quantity=qty,
                timestamp_ms=ts,
            )
        )

    def _cancel_all_pending(self) -> None:
        """撤销所有挂单"""
        self._pending_orders.clear()

    # ------------------------------------------------------------------
    # 平仓与收尾
    # ------------------------------------------------------------------

    def _force_close(self, mid_price: float) -> None:
        """Episode 结束时强制平仓

        以当前 mid_price 作为平仓价格，构造一笔 taker 成交把仓位清零。

        Args:
            mid_price: 当前中间价
        """
        assert self._account is not None

        pos_qty: int = self._account.position.quantity
        if pos_qty == 0:
            return

        # 确定平仓方向: 多头 -> 卖出, 空头 -> 买入
        close_side: int = -1 if pos_qty > 0 else 1
        close_qty: int = abs(pos_qty)

        self._process_market_fill(close_side, mid_price, close_qty)

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------

    def _find_trade_idx(self, timestamp_ms: int) -> int:
        """二分查找 >= timestamp_ms 的第一个成交索引

        Args:
            timestamp_ms: 目标时间戳

        Returns:
            第一个 >= timestamp_ms 的成交索引，
            如果所有成交都 < timestamp_ms 则返回 len(trades)
        """
        return bisect.bisect_left(self._trade_timestamps, timestamp_ms)


# ---------------------------------------------------------------------------
# 模块级辅助函数
# ---------------------------------------------------------------------------

def _round_price(price: float, tick_size: float) -> float:
    """价格取整到 tick_size 的整数倍，保证至少为一个 tick_size

    与 Agent 基类中 fast_round_price / _py_round_price 逻辑一致。

    Args:
        price: 原始价格
        tick_size: 最小变动单位

    Returns:
        取整后的价格
    """
    return max(tick_size, round(price / tick_size) * tick_size)
