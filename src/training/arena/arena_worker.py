"""Arena Worker 进程池架构核心实现

每个 Worker 进程独立运行若干竞技场的完整 tick 循环，消除 tick 级 IPC 同步。
Worker 内部持有本地 BatchNetworkCache、ArenaState、MatchingEngine，
通过 multiprocessing.Queue 与主进程通信。

通信协议：
- update_networks: 主进程 -> Worker 发送新网络参数，Worker 更新本地缓存
- run_episode: 主进程 -> Worker 发送 episode 参数，Worker 独立运行并返回适应度结果
- shutdown: 主进程 -> Worker 发送关闭命令
"""

from __future__ import annotations

import gc
import logging
import os
import random
import time
from collections import deque
from dataclasses import dataclass, field
from multiprocessing import Process, Queue
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.bio.agents.base import ActionType, AgentType
from src.config.config import Config, NoiseTraderConfig
from src.market.adl.adl_manager import ADLCandidate, ADLManager
from src.market.market_state import NormalizedMarketState
from src.market.matching.matching_engine import MatchingEngine
from src.market.matching.trade import Trade
from src.market.noise_trader.noise_trader import NoiseTrader
from src.market.orderbook.order import Order, OrderSide, OrderType
from src.training.fast_math import log_normalize_signed, log_normalize_unsigned

from .arena_state import (
    AgentAccountState,
    ArenaState,
    NoiseTraderAccountState,
    calculate_order_quantity_from_state,
    calculate_skew_factor_from_state,
)
from .execute_worker import NoiseTraderDecision, NoiseTraderTradeResult

# BatchNetworkCache 导入
CACHE_TYPE_FULL = 1
CACHE_TYPE_MARKET_MAKER = 2
try:
    from src.training._cython.batch_decide_openmp import BatchNetworkCache

    HAS_OPENMP_DECIDE = True
except ImportError:
    HAS_OPENMP_DECIDE = False
    BatchNetworkCache = None  # type: ignore

logger = logging.getLogger(__name__)

# 做市商最小订单常量（与 parallel_arena_trainer.py 保持一致）
MM_MIN_ORDER_QUANTITY: int = 1
MM_MIN_RATIO_THRESHOLD: float = 0.001


# ============================================================================
# 数据类
# ============================================================================


@dataclass
class AgentInfo:
    """Worker 进程需要的轻量级 Agent 结构信息

    包含创建 AgentAccountState 和注册到 MatchingEngine 所需的全部参数。
    在主进程创建后通过 Queue 序列化传递给 Worker。

    Attributes:
        agent_id: Agent ID
        agent_type: Agent 类型
        sub_pop_id: 子种群 ID
        network_index: 在 BatchNetworkCache 中的索引
        initial_balance: 初始余额
        leverage: 杠杆倍数
        maintenance_margin_rate: 维持保证金率
        maker_fee_rate: 挂单费率
        taker_fee_rate: 吃单费率
    """

    agent_id: int
    agent_type: AgentType
    sub_pop_id: int
    network_index: int
    initial_balance: float
    leverage: float
    maintenance_margin_rate: float
    maker_fee_rate: float
    taker_fee_rate: float


@dataclass
class ArenaEpisodeStats:
    """单个竞技场的 episode 统计信息

    Attributes:
        end_reason: 结束原因（None 表示正常运行完毕）
        end_tick: 结束时的 tick 数
        high_price: episode 最高价
        low_price: episode 最低价
    """

    end_reason: str | None
    end_tick: int
    high_price: float
    low_price: float


@dataclass
class EpisodeResult:
    """Worker 返回给主进程的结果

    Attributes:
        worker_id: Worker ID
        accumulated_fitness: 所有竞技场汇总的适应度，
            key 为 (AgentType, sub_pop_id)，value 为适应度数组
        per_arena_fitness: 每个竞技场独立的适应度（供 league trainer 使用），
            key 为 arena_id，value 为 {AgentType: fitness_array}
        arena_stats: 每个竞技场的统计信息
    """

    worker_id: int
    accumulated_fitness: dict[tuple[AgentType, int], NDArray[np.float32]]
    per_arena_fitness: dict[int, dict[AgentType, NDArray[np.float32]]]
    arena_stats: dict[int, ArenaEpisodeStats]


# ============================================================================
# 原子动作数据结构（从 execute_worker.py 复用定义逻辑）
# ============================================================================


class _AtomicActionType:
    """原子动作类型常量"""

    CANCEL = 1
    LIMIT_BUY = 2
    LIMIT_SELL = 3
    MARKET_BUY = 4
    MARKET_SELL = 5


@dataclass
class _AtomicAction:
    """原子动作（Worker 本地使用，无需 IPC 序列化）

    Attributes:
        action_type: 动作类型
        agent_id: Agent ID
        order_id: 订单 ID（CANCEL 动作使用）
        price: 订单价格（LIMIT 动作使用）
        quantity: 订单数量（非 CANCEL 动作使用）
        is_market_maker: 是否是做市商
        is_noise_trader: 是否是噪声交易者
    """

    action_type: int
    agent_id: int
    order_id: int = 0
    price: float = 0.0
    quantity: int = 0
    is_market_maker: bool = False
    is_noise_trader: bool = False


# ============================================================================
# 模块级函数：从 ParallelArenaTrainer 提取为独立函数
# ============================================================================


def check_liquidations_vectorized(
    arena: ArenaState, current_price: float
) -> list[int]:
    """向量化强平检查

    使用 NumPy 向量化操作批量计算保证金率，返回需要强平的 Agent ID 列表。

    Args:
        arena: 竞技场状态
        current_price: 当前市场价格

    Returns:
        需要强平的 Agent ID 列表
    """
    if (
        arena._balances is None
        or arena._is_liquidated_flags is None
        or arena._position_avg_prices is None
        or arena._position_quantities is None
        or arena._maintenance_margins is None
        or arena._idx_to_agent_id is None
    ):
        return []

    # 活跃 Agent 掩码
    active_mask = ~arena._is_liquidated_flags

    # 向量化计算 unrealized_pnl
    unrealized_pnl = (
        current_price - arena._position_avg_prices
    ) * arena._position_quantities

    # 向量化计算 equity
    equities = arena._balances + unrealized_pnl

    # 向量化计算 position_value
    position_values = np.abs(arena._position_quantities) * current_price

    # 向量化计算 margin_ratio
    with np.errstate(divide="ignore", invalid="ignore"):
        margin_ratios = np.where(
            position_values > 0, equities / position_values, np.inf
        )

    # 找出需要强平的
    need_liquidation = active_mask & (margin_ratios < arena._maintenance_margins)
    liquidation_indices = np.where(need_liquidation)[0]

    return [int(arena._idx_to_agent_id[idx]) for idx in liquidation_indices]


def handle_liquidations(arena: ArenaState, current_price: float) -> None:
    """处理单个竞技场的强平（三阶段：撤单 -> 市价平仓 -> ADL）

    Worker 本地直接执行，不经过 IPC。

    Args:
        arena: 竞技场状态
        current_price: 当前市场价格
    """
    # 向量化检查强平条件
    liquidation_ids = check_liquidations_vectorized(arena, current_price)
    agents_to_liquidate: list[AgentAccountState] = [
        arena.agent_states[agent_id] for agent_id in liquidation_ids
    ]

    if not agents_to_liquidate:
        return

    # 阶段1: 撤销挂单
    for agent_state in agents_to_liquidate:
        cancel_agent_orders(arena, agent_state)

    # 阶段2: 市价平仓
    agents_need_adl: list[tuple[AgentAccountState, int, bool]] = []
    for agent_state in agents_to_liquidate:
        remaining_qty, is_long = execute_liquidation(
            arena, agent_state, current_price
        )
        if remaining_qty > 0:
            agents_need_adl.append((agent_state, remaining_qty, is_long))

        agent_state.is_liquidated = True
        arena.mark_agent_liquidated(agent_state.agent_id, agent_state.agent_type)

        if agent_state.balance < 0:
            agent_state.balance = 0.0

        arena.sync_state_to_array(agent_state.agent_id)

    # 阶段3: ADL
    if agents_need_adl:
        latest_price = arena.matching_engine._orderbook.last_price
        execute_adl(arena, agents_need_adl, latest_price)


def cancel_agent_orders(arena: ArenaState, agent_state: AgentAccountState) -> None:
    """撤销 Agent 在竞技场中的挂单

    Args:
        arena: 竞技场状态
        agent_state: Agent 账户状态
    """
    if agent_state.agent_type == AgentType.MARKET_MAKER:
        for order_id in agent_state.bid_order_ids + agent_state.ask_order_ids:
            arena.matching_engine.cancel_order(order_id)
        agent_state.bid_order_ids.clear()
        agent_state.ask_order_ids.clear()
    else:
        if agent_state.pending_order_id is not None:
            arena.matching_engine.cancel_order(agent_state.pending_order_id)
            agent_state.pending_order_id = None


def execute_liquidation(
    arena: ArenaState,
    agent_state: AgentAccountState,
    current_price: float,
) -> tuple[int, bool]:
    """在竞技场中执行强平市价单

    Args:
        arena: 竞技场状态
        agent_state: Agent 账户状态
        current_price: 当前市场价格

    Returns:
        (remaining_qty, is_long): 未成交的剩余数量和持仓方向
    """
    position_qty = agent_state.position_quantity
    if position_qty == 0:
        return 0, True

    is_long = position_qty > 0
    target_qty = abs(position_qty)

    side = OrderSide.SELL if is_long else OrderSide.BUY
    order_id = agent_state.generate_order_id(arena.arena_id)
    order = Order(
        order_id=order_id,
        agent_id=agent_state.agent_id,
        side=side,
        order_type=OrderType.MARKET,
        price=0.0,
        quantity=target_qty,
    )

    trades = arena.matching_engine.process_order(order)

    for trade in trades:
        is_buyer = trade.is_buyer_taker
        fee = trade.buyer_fee if trade.is_buyer_taker else trade.seller_fee
        agent_state.on_trade(trade.price, trade.quantity, is_buyer, fee, False)
        arena.sync_state_to_array(agent_state.agent_id)
        arena.recent_trades.append(trade)

        maker_id = trade.seller_id if trade.is_buyer_taker else trade.buyer_id
        maker_state = arena.agent_states.get(maker_id)
        if maker_state is not None:
            maker_is_buyer = not trade.is_buyer_taker
            maker_fee = (
                trade.seller_fee if trade.is_buyer_taker else trade.buyer_fee
            )
            maker_state.on_trade(
                trade.price, trade.quantity, maker_is_buyer, maker_fee, True
            )
            arena.sync_state_to_array(maker_id)
        else:
            # 可能是噪声交易者
            nt_state = arena.noise_trader_states.get(maker_id)
            if nt_state is not None:
                nt_state.on_trade(
                    trade.price, trade.quantity, not trade.is_buyer_taker
                )

    remaining_qty = abs(agent_state.position_quantity)
    return remaining_qty, is_long


def execute_adl(
    arena: ArenaState,
    agents_need_adl: list[tuple[AgentAccountState, int, bool]],
    current_price: float,
) -> None:
    """在竞技场中执行 ADL

    Args:
        arena: 竞技场状态
        agents_need_adl: 需要 ADL 的 Agent 列表，
            格式: [(agent_state, remaining_qty, is_long), ...]
        current_price: 当前市场价格
    """
    adl_price = arena.adl_manager.get_adl_price(current_price)

    # 计算 ADL 候选
    long_candidates: list[ADLCandidate] = []
    short_candidates: list[ADLCandidate] = []

    for agent_state in arena.agent_states.values():
        if agent_state.is_liquidated:
            continue
        if agent_state.position_quantity == 0:
            continue

        equity = agent_state.get_equity(current_price)
        pnl_percent = (
            equity - agent_state.initial_balance
        ) / agent_state.initial_balance

        if pnl_percent <= 0:
            continue

        position_value = abs(agent_state.position_quantity) * current_price
        effective_leverage = position_value / equity if equity > 0 else 0.0
        adl_score = pnl_percent * effective_leverage

        # 创建模拟的 participant（用于 ADLCandidate 接口兼容）
        class MockParticipant:
            def __init__(self, state: AgentAccountState) -> None:
                self._state = state
                self.account = self

            @property
            def position(self) -> "MockParticipant":
                return self

            @property
            def quantity(self) -> int:
                return self._state.position_quantity

        candidate = ADLCandidate(
            participant=MockParticipant(agent_state),  # type: ignore
            position_qty=agent_state.position_quantity,
            pnl_percent=pnl_percent,
            effective_leverage=effective_leverage,
            adl_score=adl_score,
        )

        if agent_state.position_quantity > 0:
            long_candidates.append(candidate)
        else:
            short_candidates.append(candidate)

    long_candidates.sort(key=lambda c: c.adl_score, reverse=True)
    short_candidates.sort(key=lambda c: c.adl_score, reverse=True)

    # 执行 ADL
    for agent_state, remaining_qty, is_long in agents_need_adl:
        candidates = short_candidates if is_long else long_candidates

        for candidate in candidates:
            if remaining_qty <= 0:
                break

            candidate_available_qty = abs(candidate.position_qty)
            trade_qty = min(candidate_available_qty, remaining_qty)

            if trade_qty <= 0:
                continue

            # 更新被强平方
            if is_long:
                agent_state.position_quantity -= trade_qty
            else:
                agent_state.position_quantity += trade_qty
            agent_state.balance += (
                (adl_price - agent_state.position_avg_price)
                * trade_qty
                * (1 if is_long else -1)
            )
            arena.sync_state_to_array(agent_state.agent_id)

            # 更新对手方
            if candidate.position_qty > 0:
                candidate.position_qty -= trade_qty
            else:
                candidate.position_qty += trade_qty
            # 对手方是 mock 对象，通过 _state 同步
            counter_state: AgentAccountState = candidate.participant._state  # type: ignore
            # 同步对手方的持仓数量（ADL 直接修改了 candidate.position_qty）
            counter_state.position_quantity = candidate.position_qty
            arena.sync_state_to_array(counter_state.agent_id)

            remaining_qty -= trade_qty

        # 兜底处理
        if agent_state.position_quantity != 0:
            agent_state.position_quantity = 0
            agent_state.position_avg_price = 0.0
            arena.sync_state_to_array(agent_state.agent_id)


def update_trade_accounts(
    arena: ArenaState,
    agent_state: AgentAccountState,
    trades: list[Trade],
) -> None:
    """更新成交相关的账户状态

    处理 taker 和 maker 双方的账户更新。

    Args:
        arena: 竞技场状态
        agent_state: taker 的 Agent 账户状态
        trades: 成交列表
    """
    for trade in trades:
        is_buyer = trade.is_buyer_taker
        fee = trade.buyer_fee if is_buyer else trade.seller_fee
        agent_state.on_trade(
            trade.price, trade.quantity, is_buyer, fee, is_maker=False
        )
        arena.sync_state_to_array(agent_state.agent_id)
        arena.recent_trades.append(trade)

        # 更新 maker 账户
        maker_id = trade.seller_id if trade.is_buyer_taker else trade.buyer_id
        maker_state = arena.agent_states.get(maker_id)
        if maker_state is not None:
            maker_is_buyer = not trade.is_buyer_taker
            maker_fee = (
                trade.seller_fee if trade.is_buyer_taker else trade.buyer_fee
            )
            maker_state.on_trade(
                trade.price,
                trade.quantity,
                maker_is_buyer,
                maker_fee,
                is_maker=True,
            )
            arena.sync_state_to_array(maker_id)
        else:
            # 可能是噪声交易者
            nt_state = arena.noise_trader_states.get(maker_id)
            if nt_state is not None:
                nt_state.on_trade(
                    trade.price, trade.quantity, not trade.is_buyer_taker
                )


def compute_noise_trader_decisions(
    arena: ArenaState, noise_trader_config: NoiseTraderConfig
) -> list[NoiseTraderDecision]:
    """计算噪声交易者决策

    调用 NoiseTraderAccountState.decide() 获取决策，转换为 NoiseTraderDecision 格式。

    Args:
        arena: 竞技场状态
        noise_trader_config: 噪声交易者配置

    Returns:
        噪声交易者决策列表（仅包含需要行动的噪声交易者）
    """
    decisions: list[NoiseTraderDecision] = []

    if not arena.noise_trader_states:
        return decisions

    for nt_state in arena.noise_trader_states.values():
        should_act, direction, quantity = nt_state.decide(
            noise_trader_config.action_probability
        )

        if should_act and direction != 0 and quantity > 0:
            decisions.append(
                NoiseTraderDecision(
                    trader_id=nt_state.trader_id,
                    direction=direction,
                    quantity=quantity,
                )
            )

    return decisions


def compute_market_state(
    arena: ArenaState,
    buffers: dict[str, NDArray[np.float32]],
    ema_alpha: float,
) -> NormalizedMarketState:
    """计算单个竞技场的归一化市场状态

    Worker 本地版本：直接使用 arena 的 OrderBook，不依赖 _worker_depth_cache。

    Args:
        arena: 竞技场状态
        buffers: 预分配的市场状态缓冲区
        ema_alpha: EMA 平滑系数

    Returns:
        归一化后的市场状态
    """
    orderbook = arena.matching_engine._orderbook

    # 获取实时参考价格
    current_mid_price = orderbook.get_mid_price()
    if current_mid_price is None:
        current_mid_price = orderbook.last_price
    if current_mid_price == 0:
        current_mid_price = 100.0

    # 更新 EMA 平滑价格
    arena.smooth_mid_price = (
        ema_alpha * current_mid_price
        + (1 - ema_alpha) * arena.smooth_mid_price
    )
    smooth_mid_price = arena.smooth_mid_price

    tick_size = orderbook.tick_size

    # 直接从本地 orderbook 获取深度
    bid_depth, ask_depth = orderbook.get_depth_numpy(levels=100)

    # 获取并清零缓冲区
    bid_data = buffers["bid_data"]
    ask_data = buffers["ask_data"]
    trade_prices = buffers["trade_prices"]
    trade_quantities = buffers["trade_quantities"]
    tick_prices_normalized = buffers["tick_prices"]
    tick_volumes_normalized = buffers["tick_volumes"]
    tick_amounts_normalized = buffers["tick_amounts"]

    bid_data.fill(0)
    ask_data.fill(0)
    trade_prices.fill(0)
    trade_quantities.fill(0)
    tick_prices_normalized.fill(0)
    tick_volumes_normalized.fill(0)
    tick_amounts_normalized.fill(0)

    # 向量化买盘
    bid_prices = bid_depth[:, 0]
    bid_qtys = bid_depth[:, 1]
    bid_valid_mask = bid_prices > 0
    n_bids = int(np.sum(bid_valid_mask))
    if n_bids > 0 and smooth_mid_price > 0:
        bid_data[0: n_bids * 2: 2] = (
            bid_prices[:n_bids] - smooth_mid_price
        ) / smooth_mid_price
        bid_data[1: n_bids * 2: 2] = log_normalize_unsigned(bid_qtys[:n_bids])

    # 向量化卖盘
    ask_prices = ask_depth[:, 0]
    ask_qtys = ask_depth[:, 1]
    ask_valid_mask = ask_prices > 0
    n_asks = int(np.sum(ask_valid_mask))
    if n_asks > 0 and smooth_mid_price > 0:
        ask_data[0: n_asks * 2: 2] = (
            ask_prices[:n_asks] - smooth_mid_price
        ) / smooth_mid_price
        ask_data[1: n_asks * 2: 2] = log_normalize_unsigned(ask_qtys[:n_asks])

    # 向量化成交
    n_trades = len(arena.recent_trades)
    if n_trades > 0:
        prices_arr = np.empty(n_trades, dtype=np.float32)
        qtys_arr = np.empty(n_trades, dtype=np.float32)
        for i, t in enumerate(arena.recent_trades):
            prices_arr[i] = t.price
            qtys_arr[i] = t.quantity if t.is_buyer_taker else -t.quantity

        if smooth_mid_price > 0:
            trade_prices[:n_trades] = (
                prices_arr - smooth_mid_price
            ) / smooth_mid_price
        trade_quantities[:n_trades] = log_normalize_signed(qtys_arr)

    # Tick 历史价格归一化
    if arena.tick_history_prices:
        hist_prices = np.array(arena.tick_history_prices, dtype=np.float32)
        volumes = np.array(arena.tick_history_volumes, dtype=np.float32)
        amounts = np.array(arena.tick_history_amounts, dtype=np.float32)
        n = len(hist_prices)

        base_price = hist_prices[0]
        if base_price > 0:
            tick_prices_normalized[-n:] = (hist_prices - base_price) / base_price

        tick_volumes_normalized[-n:] = log_normalize_signed(volumes)
        tick_amounts_normalized[-n:] = log_normalize_signed(amounts, scale=12.0)

    return NormalizedMarketState(
        mid_price=smooth_mid_price,
        tick_size=tick_size,
        bid_data=bid_data.copy(),
        ask_data=ask_data.copy(),
        trade_prices=trade_prices.copy(),
        trade_quantities=trade_quantities.copy(),
        tick_history_prices=tick_prices_normalized.copy(),
        tick_history_volumes=tick_volumes_normalized.copy(),
        tick_history_amounts=tick_amounts_normalized.copy(),
    )


def check_early_end(
    arena: ArenaState, pop_total_counts: dict[AgentType, int]
) -> tuple[str, AgentType | None] | None:
    """检查单个竞技场是否应该提前结束

    Worker 本地版本：直接使用 arena 的 OrderBook 检查。

    Args:
        arena: 竞技场状态
        pop_total_counts: 每种类型的总 Agent 数量

    Returns:
        (reason, agent_type) 或 None（正常继续）
    """
    # 检查种群存活数量
    for agent_type, total in pop_total_counts.items():
        if total > 0:
            liquidated = arena.pop_liquidated_counts.get(agent_type, 0)
            alive = total - liquidated
            if alive < total / 4:
                return ("population_depleted", agent_type)

    # 检查订单簿单边挂单（直接查本地 orderbook）
    orderbook = arena.matching_engine._orderbook
    has_bids = orderbook.get_best_bid() is not None
    has_asks = orderbook.get_best_ask() is not None

    if has_bids != has_asks:
        return ("one_sided_orderbook", None)

    return None


def aggregate_tick_trades(tick_trades: list[Trade]) -> tuple[float, float]:
    """聚合本 tick 的成交量和成交额

    Args:
        tick_trades: 本 tick 的成交列表

    Returns:
        (volume, amount): 带方向的成交量和成交额
    """
    if not tick_trades:
        return 0.0, 0.0

    buy_volume = sum(t.quantity for t in tick_trades if t.is_buyer_taker)
    sell_volume = sum(t.quantity for t in tick_trades if not t.is_buyer_taker)
    buy_amount = sum(t.price * t.quantity for t in tick_trades if t.is_buyer_taker)
    sell_amount = sum(
        t.price * t.quantity for t in tick_trades if not t.is_buyer_taker
    )

    total_volume = buy_volume + sell_volume
    total_amount = buy_amount + sell_amount

    if buy_amount > sell_amount:
        return float(total_volume), total_amount
    elif sell_amount > buy_amount:
        return float(-total_volume), -total_amount
    return 0.0, 0.0


def update_episode_price_stats_from_trades(
    arena: ArenaState,
    tick_trades: list[Trade],
    fallback_price: float | None = None,
) -> None:
    """使用本 tick 成交价格更新 episode high/low

    Args:
        arena: 竞技场状态
        tick_trades: 本 tick 的成交列表
        fallback_price: 无成交时的回退价格
    """
    if tick_trades:
        tick_high = max(trade.price for trade in tick_trades)
        tick_low = min(trade.price for trade in tick_trades)
    elif fallback_price is not None:
        tick_high = fallback_price
        tick_low = fallback_price
    else:
        return

    if tick_high > arena.episode_high_price:
        arena.episode_high_price = tick_high
    if tick_low < arena.episode_low_price:
        arena.episode_low_price = tick_low


def execute_tick_local(
    arena: ArenaState,
    retail_decisions: NDArray[np.float64] | None,
    retail_agent_ids: NDArray[np.int64] | None,
    mm_decisions: NDArray[np.float64] | None,
    mm_agent_ids: NDArray[np.int64] | None,
    noise_decisions: list[NoiseTraderDecision],
    liquidated_agents: list[tuple[int, int, bool]],
) -> list[Trade]:
    """在 Worker 本地执行一个 tick 的所有订单

    参考 execute_worker.py 的 _handle_execute 中的原子动作模式，
    但直接操作 ArenaState 而不需要 IPC。

    执行流程：
    1. 强平处理（不打乱，优先执行）
    2. 收集原子动作（noise_decisions, mm_decisions, retail_decisions）
    3. 随机打乱原子动作
    4. 逐个执行原子动作
    5. 返回所有成交

    Args:
        arena: 竞技场状态
        retail_decisions: 非 MM 决策数组 shape (N, 4)，
            列顺序 [action_type, side, price, quantity]
        retail_agent_ids: 非 MM Agent ID 数组 shape (N,)
        mm_decisions: MM 决策数组 shape (M, 42)，
            列顺序 [num_bid, num_ask, bid_prices[10], bid_qtys[10],
                    ask_prices[10], ask_qtys[10]]
        mm_agent_ids: MM Agent ID 数组 shape (M,)
        noise_decisions: 噪声交易者决策列表
        liquidated_agents: 强平 Agent 列表，格式 (agent_id, position_qty, is_mm)

    Returns:
        所有成交的 Trade 列表
    """
    all_trades: list[Trade] = []
    matching_engine = arena.matching_engine
    orderbook = matching_engine._orderbook
    process_order = matching_engine.process_order
    cancel_order = matching_engine.cancel_order

    # ====== 第一部分：强平处理（不参与打乱，优先执行）======
    for agent_id, position_qty, is_mm in liquidated_agents:
        # 撤单
        agent_state = arena.agent_states.get(agent_id)
        if agent_state is not None:
            if is_mm:
                for oid in agent_state.bid_order_ids:
                    cancel_order(oid)
                for oid in agent_state.ask_order_ids:
                    cancel_order(oid)
                agent_state.bid_order_ids.clear()
                agent_state.ask_order_ids.clear()
            else:
                if agent_state.pending_order_id is not None:
                    cancel_order(agent_state.pending_order_id)
                    agent_state.pending_order_id = None

        # 市价平仓
        if position_qty != 0:
            if agent_state is not None:
                order_id = agent_state.generate_order_id(arena.arena_id)
            else:
                order_id = (arena.arena_id << 48) | (agent_id << 16) | 0xFFFF

            side = OrderSide.SELL if position_qty > 0 else OrderSide.BUY
            order = Order(
                order_id=order_id,
                agent_id=agent_id,
                side=side,
                order_type=OrderType.MARKET,
                price=0.0,
                quantity=abs(position_qty),
            )
            trades = process_order(order)
            for trade in trades:
                all_trades.append(trade)
                arena.recent_trades.append(trade)
                _update_trade_participants(arena, trade)

    # ====== 第二部分：收集所有原子动作 ======
    atomic_actions: list[_AtomicAction] = []

    # 2.1 收集噪声交易者动作
    for decision in noise_decisions:
        if decision.direction == 0 or decision.quantity <= 0:
            continue
        if decision.direction > 0:
            atomic_actions.append(
                _AtomicAction(
                    _AtomicActionType.MARKET_BUY,
                    decision.trader_id,
                    quantity=decision.quantity,
                    is_noise_trader=True,
                )
            )
        else:
            atomic_actions.append(
                _AtomicAction(
                    _AtomicActionType.MARKET_SELL,
                    decision.trader_id,
                    quantity=decision.quantity,
                    is_noise_trader=True,
                )
            )

    # 2.2 收集做市商动作（撤旧单 + 挂新单）
    if mm_decisions is not None and mm_agent_ids is not None and len(mm_decisions) > 0:
        for row_idx in range(len(mm_decisions)):
            agent_id = int(mm_agent_ids[row_idx])
            num_bid = int(mm_decisions[row_idx, 0])
            num_ask = int(mm_decisions[row_idx, 1])

            # 撤旧单
            mm_state = arena.agent_states.get(agent_id)
            if mm_state is not None:
                for oid in mm_state.bid_order_ids:
                    atomic_actions.append(
                        _AtomicAction(
                            _AtomicActionType.CANCEL,
                            agent_id,
                            order_id=oid,
                            is_market_maker=True,
                        )
                    )
                for oid in mm_state.ask_order_ids:
                    atomic_actions.append(
                        _AtomicAction(
                            _AtomicActionType.CANCEL,
                            agent_id,
                            order_id=oid,
                            is_market_maker=True,
                        )
                    )
                # 清空旧挂单列表
                mm_state.bid_order_ids = []
                mm_state.ask_order_ids = []

            # 挂新买单（价格在列 2-11，数量在列 12-21）
            for k in range(num_bid):
                price = float(mm_decisions[row_idx, 2 + k])
                qty = int(mm_decisions[row_idx, 12 + k])
                if qty > 0:
                    atomic_actions.append(
                        _AtomicAction(
                            _AtomicActionType.LIMIT_BUY,
                            agent_id,
                            price=price,
                            quantity=qty,
                            is_market_maker=True,
                        )
                    )
            # 挂新卖单（价格在列 22-31，数量在列 32-41）
            for k in range(num_ask):
                price = float(mm_decisions[row_idx, 22 + k])
                qty = int(mm_decisions[row_idx, 32 + k])
                if qty > 0:
                    atomic_actions.append(
                        _AtomicAction(
                            _AtomicActionType.LIMIT_SELL,
                            agent_id,
                            price=price,
                            quantity=qty,
                            is_market_maker=True,
                        )
                    )

    # 2.3 收集非做市商动作
    if (
        retail_decisions is not None
        and retail_agent_ids is not None
        and len(retail_decisions) > 0
    ):
        for i in range(len(retail_decisions)):
            agent_id = int(retail_agent_ids[i])
            action_int = int(retail_decisions[i, 0])
            price = float(retail_decisions[i, 2])
            quantity = int(retail_decisions[i, 3])

            if action_int == 0:  # HOLD
                continue
            elif action_int in (1, 2):  # PLACE_BID / PLACE_ASK
                # 先撤旧单
                r_state = arena.agent_states.get(agent_id)
                if r_state is not None and r_state.pending_order_id is not None:
                    atomic_actions.append(
                        _AtomicAction(
                            _AtomicActionType.CANCEL,
                            agent_id,
                            order_id=r_state.pending_order_id,
                        )
                    )
                    r_state.pending_order_id = None
                # 挂新单
                a_type = (
                    _AtomicActionType.LIMIT_BUY
                    if action_int == 1
                    else _AtomicActionType.LIMIT_SELL
                )
                atomic_actions.append(
                    _AtomicAction(a_type, agent_id, price=price, quantity=quantity)
                )
            elif action_int == 3:  # CANCEL
                r_state = arena.agent_states.get(agent_id)
                if r_state is not None and r_state.pending_order_id is not None:
                    atomic_actions.append(
                        _AtomicAction(
                            _AtomicActionType.CANCEL,
                            agent_id,
                            order_id=r_state.pending_order_id,
                        )
                    )
                    r_state.pending_order_id = None
            elif action_int in (4, 5):  # MARKET_BUY / MARKET_SELL
                a_type = (
                    _AtomicActionType.MARKET_BUY
                    if action_int == 4
                    else _AtomicActionType.MARKET_SELL
                )
                atomic_actions.append(
                    _AtomicAction(a_type, agent_id, quantity=quantity)
                )

    # ====== 第三部分：随机打乱并执行 ======
    random.shuffle(atomic_actions)

    for action in atomic_actions:
        _execute_atomic_action_local(arena, action, all_trades)

    return all_trades


def _execute_atomic_action_local(
    arena: ArenaState,
    action: _AtomicAction,
    all_trades: list[Trade],
) -> None:
    """执行单个原子动作（Worker 本地版本）

    直接操作 arena.matching_engine 和更新 arena.agent_states。

    Args:
        arena: 竞技场状态
        action: 原子动作
        all_trades: 所有成交列表（会被修改）
    """
    matching_engine = arena.matching_engine
    orderbook = matching_engine._orderbook
    process_order = matching_engine.process_order
    cancel_order_fn = matching_engine.cancel_order
    order_map_get = orderbook.order_map.get

    agent_id = action.agent_id

    if action.action_type == _AtomicActionType.CANCEL:
        order_id = action.order_id
        if order_id != 0:
            cancel_order_fn(order_id)

            # 更新状态
            if action.is_market_maker:
                mm_state = arena.agent_states.get(agent_id)
                if mm_state is not None:
                    if order_id in mm_state.bid_order_ids:
                        mm_state.bid_order_ids.remove(order_id)
                    if order_id in mm_state.ask_order_ids:
                        mm_state.ask_order_ids.remove(order_id)
            elif not action.is_noise_trader:
                r_state = arena.agent_states.get(agent_id)
                if r_state is not None and r_state.pending_order_id == order_id:
                    r_state.pending_order_id = None

    elif action.action_type == _AtomicActionType.LIMIT_BUY:
        agent_state = arena.agent_states.get(agent_id)
        if agent_state is not None:
            order_id = agent_state.generate_order_id(arena.arena_id)
        else:
            # 噪声交易者
            nt_state = arena.noise_trader_states.get(agent_id)
            if nt_state is not None:
                order_id = nt_state.generate_order_id(arena.arena_id)
            else:
                return

        order = Order(
            order_id=order_id,
            agent_id=agent_id,
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            price=action.price,
            quantity=action.quantity,
        )
        trades = process_order(order)
        for trade in trades:
            all_trades.append(trade)
            arena.recent_trades.append(trade)
            _update_trade_participants(arena, trade)

        # 更新挂单状态
        if order_map_get(order_id):
            if action.is_market_maker:
                mm_state = arena.agent_states.get(agent_id)
                if mm_state is not None:
                    mm_state.bid_order_ids.append(order_id)
            elif not action.is_noise_trader:
                r_state = arena.agent_states.get(agent_id)
                if r_state is not None:
                    r_state.pending_order_id = order_id
        elif not action.is_market_maker and not action.is_noise_trader:
            r_state = arena.agent_states.get(agent_id)
            if r_state is not None:
                r_state.pending_order_id = None

    elif action.action_type == _AtomicActionType.LIMIT_SELL:
        agent_state = arena.agent_states.get(agent_id)
        if agent_state is not None:
            order_id = agent_state.generate_order_id(arena.arena_id)
        else:
            nt_state = arena.noise_trader_states.get(agent_id)
            if nt_state is not None:
                order_id = nt_state.generate_order_id(arena.arena_id)
            else:
                return

        order = Order(
            order_id=order_id,
            agent_id=agent_id,
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            price=action.price,
            quantity=action.quantity,
        )
        trades = process_order(order)
        for trade in trades:
            all_trades.append(trade)
            arena.recent_trades.append(trade)
            _update_trade_participants(arena, trade)

        # 更新挂单状态
        if order_map_get(order_id):
            if action.is_market_maker:
                mm_state = arena.agent_states.get(agent_id)
                if mm_state is not None:
                    mm_state.ask_order_ids.append(order_id)
            elif not action.is_noise_trader:
                r_state = arena.agent_states.get(agent_id)
                if r_state is not None:
                    r_state.pending_order_id = order_id
        elif not action.is_market_maker and not action.is_noise_trader:
            r_state = arena.agent_states.get(agent_id)
            if r_state is not None:
                r_state.pending_order_id = None

    elif action.action_type == _AtomicActionType.MARKET_BUY:
        if action.is_noise_trader:
            nt_state = arena.noise_trader_states.get(agent_id)
            if nt_state is not None:
                order_id = nt_state.generate_order_id(arena.arena_id)
            else:
                return
        else:
            agent_state = arena.agent_states.get(agent_id)
            if agent_state is not None:
                order_id = agent_state.generate_order_id(arena.arena_id)
            else:
                return

        order = Order(
            order_id=order_id,
            agent_id=agent_id,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            price=0.0,
            quantity=action.quantity,
        )
        trades = process_order(order)
        for trade in trades:
            all_trades.append(trade)
            arena.recent_trades.append(trade)
            _update_trade_participants(arena, trade)

    elif action.action_type == _AtomicActionType.MARKET_SELL:
        if action.is_noise_trader:
            nt_state = arena.noise_trader_states.get(agent_id)
            if nt_state is not None:
                order_id = nt_state.generate_order_id(arena.arena_id)
            else:
                return
        else:
            agent_state = arena.agent_states.get(agent_id)
            if agent_state is not None:
                order_id = agent_state.generate_order_id(arena.arena_id)
            else:
                return

        order = Order(
            order_id=order_id,
            agent_id=agent_id,
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            price=0.0,
            quantity=action.quantity,
        )
        trades = process_order(order)
        for trade in trades:
            all_trades.append(trade)
            arena.recent_trades.append(trade)
            _update_trade_participants(arena, trade)


def _update_trade_participants(arena: ArenaState, trade: Trade) -> None:
    """更新成交双方的账户状态

    对每笔成交，分别更新 taker 和 maker 的 AgentAccountState 或 NoiseTraderAccountState。

    Args:
        arena: 竞技场状态
        trade: 成交对象
    """
    # 确定 taker 和 maker
    if trade.is_buyer_taker:
        taker_id = trade.buyer_id
        maker_id = trade.seller_id
        taker_fee = trade.buyer_fee
        maker_fee = trade.seller_fee
        taker_is_buyer = True
    else:
        taker_id = trade.seller_id
        maker_id = trade.buyer_id
        taker_fee = trade.seller_fee
        maker_fee = trade.buyer_fee
        taker_is_buyer = False

    # 更新 taker
    taker_state = arena.agent_states.get(taker_id)
    if taker_state is not None:
        taker_state.on_trade(
            trade.price, trade.quantity, taker_is_buyer, taker_fee, is_maker=False
        )
        arena.sync_state_to_array(taker_id)
    else:
        nt_taker = arena.noise_trader_states.get(taker_id)
        if nt_taker is not None:
            nt_taker.on_trade(trade.price, trade.quantity, taker_is_buyer)

    # 更新 maker
    maker_state = arena.agent_states.get(maker_id)
    if maker_state is not None:
        maker_state.on_trade(
            trade.price, trade.quantity, not taker_is_buyer, maker_fee, is_maker=True
        )
        arena.sync_state_to_array(maker_id)
    else:
        nt_maker = arena.noise_trader_states.get(maker_id)
        if nt_maker is not None:
            nt_maker.on_trade(trade.price, trade.quantity, not taker_is_buyer)


# ============================================================================
# Worker 本地辅助函数
# ============================================================================


def _create_worker_caches(
    config: Config, agent_infos: list[AgentInfo]
) -> dict[AgentType, Any]:
    """创建 Worker 本地的 BatchNetworkCache

    Args:
        config: 全局配置
        agent_infos: Agent 信息列表

    Returns:
        每种 Agent 类型的 BatchNetworkCache
    """
    if not HAS_OPENMP_DECIDE:
        return {}

    caches: dict[AgentType, Any] = {}
    # 统计每种类型的 agent 数量
    type_counts: dict[AgentType, int] = {}
    for info in agent_infos:
        type_counts[info.agent_type] = type_counts.get(info.agent_type, 0) + 1

    for agent_type, count in type_counts.items():
        cache_type = (
            CACHE_TYPE_MARKET_MAKER
            if agent_type == AgentType.MARKET_MAKER
            else CACHE_TYPE_FULL
        )
        caches[agent_type] = BatchNetworkCache(count, cache_type, 1)  # OMP_THREADS=1

    return caches


def _create_worker_arena_states(
    arena_ids: list[int], config: Config, agent_infos: list[AgentInfo]
) -> list[ArenaState]:
    """创建 Worker 本地的 ArenaState

    Args:
        arena_ids: 竞技场 ID 列表
        config: 全局配置
        agent_infos: Agent 信息列表

    Returns:
        ArenaState 列表
    """
    states: list[ArenaState] = []
    initial_price = config.market.initial_price
    noise_trader_config = config.noise_trader

    for arena_id in arena_ids:
        matching_engine = MatchingEngine(config.market)
        adl_manager = ADLManager()

        # 从 agent_infos 创建 AgentAccountState
        agent_states: dict[int, AgentAccountState] = {}
        for info in agent_infos:
            state = AgentAccountState(
                agent_id=info.agent_id,
                agent_type=info.agent_type,
                balance=info.initial_balance,
                position_quantity=0,
                position_avg_price=0.0,
                realized_pnl=0.0,
                leverage=info.leverage,
                maintenance_margin_rate=info.maintenance_margin_rate,
                initial_balance=info.initial_balance,
                pending_order_id=None,
                maker_volume=0,
                volatility_contribution=0.0,
                is_liquidated=False,
                order_counter=0,
                maker_fee_rate=info.maker_fee_rate,
                taker_fee_rate=info.taker_fee_rate,
                bid_order_ids=[],
                ask_order_ids=[],
                cumulative_spread_score=0.0,
                quote_tick_count=0,
            )
            agent_states[info.agent_id] = state
            matching_engine.register_agent(
                info.agent_id, info.maker_fee_rate, info.taker_fee_rate
            )

        # 创建噪声交易者状态
        noise_trader_states: dict[int, NoiseTraderAccountState] = {}
        for i in range(noise_trader_config.count):
            trader_id = -(i + 1)
            nt = NoiseTrader(trader_id, noise_trader_config)
            nt_state = NoiseTraderAccountState.from_noise_trader(nt)
            noise_trader_states[trader_id] = nt_state
            matching_engine.register_agent(trader_id, 0.0, 0.0)

        arena = ArenaState(
            arena_id=arena_id,
            matching_engine=matching_engine,
            adl_manager=adl_manager,
            agent_states=agent_states,
            noise_trader_states=noise_trader_states,
            recent_trades=deque(maxlen=100),
            price_history=deque([initial_price], maxlen=1000),
            tick_history_prices=deque([initial_price], maxlen=100),
            tick_history_volumes=deque([0.0], maxlen=100),
            tick_history_amounts=deque([0.0], maxlen=100),
            smooth_mid_price=initial_price,
            tick=0,
            pop_liquidated_counts={at: 0 for at in AgentType},
            eliminating_agents=set(),
            episode_high_price=initial_price,
            episode_low_price=initial_price,
        )
        arena.init_flat_arrays()
        states.append(arena)

    return states


def _build_worker_type_groups(
    agent_infos: list[AgentInfo],
) -> dict[AgentType, tuple[list[int], NDArray[np.int32]]]:
    """构建按类型分组的 agent 信息（不变缓存）

    Args:
        agent_infos: Agent 信息列表

    Returns:
        按 AgentType 分组的 (agent_id_list, network_index_array)
    """
    groups: dict[AgentType, tuple[list[int], list[int]]] = {}
    for info in agent_infos:
        if info.agent_type not in groups:
            groups[info.agent_type] = ([], [])
        groups[info.agent_type][0].append(info.agent_id)
        groups[info.agent_type][1].append(info.network_index)

    result: dict[AgentType, tuple[list[int], NDArray[np.int32]]] = {}
    for at, (ids, indices) in groups.items():
        result[at] = (ids, np.array(indices, dtype=np.int32))
    return result


def _create_market_state_buffers() -> dict[str, NDArray[np.float32]]:
    """创建市场状态缓冲区

    Returns:
        预分配的缓冲区字典
    """
    return {
        "bid_data": np.zeros(200, dtype=np.float32),
        "ask_data": np.zeros(200, dtype=np.float32),
        "trade_prices": np.zeros(100, dtype=np.float32),
        "trade_quantities": np.zeros(100, dtype=np.float32),
        "tick_prices": np.zeros(100, dtype=np.float32),
        "tick_volumes": np.zeros(100, dtype=np.float32),
        "tick_amounts": np.zeros(100, dtype=np.float32),
    }


def _compute_pop_total_counts(agent_infos: list[AgentInfo]) -> dict[AgentType, int]:
    """计算每种类型的总 agent 数量

    Args:
        agent_infos: Agent 信息列表

    Returns:
        每种类型的总数量
    """
    counts: dict[AgentType, int] = {}
    for info in agent_infos:
        counts[info.agent_type] = counts.get(info.agent_type, 0) + 1
    return counts


def _group_agent_infos_by_sub_pop(
    agent_infos: list[AgentInfo],
) -> dict[tuple[AgentType, int], list[AgentInfo]]:
    """按 (agent_type, sub_pop_id) 分组

    Args:
        agent_infos: Agent 信息列表

    Returns:
        分组后的字典
    """
    groups: dict[tuple[AgentType, int], list[AgentInfo]] = {}
    for info in agent_infos:
        key = (info.agent_type, info.sub_pop_id)
        if key not in groups:
            groups[key] = []
        groups[key].append(info)
    return groups


# ============================================================================
# 竞技场重置
# ============================================================================


def _reset_arena(
    arena: ArenaState,
    config: Config,
    type_groups: dict[AgentType, tuple[list[int], NDArray[np.int32]]],
    agent_infos: list[AgentInfo],
) -> None:
    """重置单个竞技场

    Args:
        arena: 竞技场状态
        config: 全局配置
        type_groups: Agent 类型分组
        agent_infos: Agent 信息列表（用于获取配置参数）
    """
    initial_price = config.market.initial_price

    # 重置订单簿
    arena.matching_engine._orderbook.clear(reset_price=initial_price)

    # 重置竞技场状态
    arena.reset_episode(initial_price)

    # 重置 Agent 账户状态
    # 构建快速查找表
    agent_config_map: dict[AgentType, dict[str, float]] = {}
    for info in agent_infos:
        if info.agent_type not in agent_config_map:
            agent_config_map[info.agent_type] = {
                "initial_balance": info.initial_balance,
                "leverage": info.leverage,
                "maintenance_margin_rate": info.maintenance_margin_rate,
                "maker_fee_rate": info.maker_fee_rate,
                "taker_fee_rate": info.taker_fee_rate,
            }

    for agent_id, state in arena.agent_states.items():
        cfg = agent_config_map.get(state.agent_type)
        if cfg is not None:
            state.balance = cfg["initial_balance"]
            state.initial_balance = cfg["initial_balance"]
            state.position_quantity = 0
            state.position_avg_price = 0.0
            state.realized_pnl = 0.0
            state.leverage = cfg["leverage"]
            state.maintenance_margin_rate = cfg["maintenance_margin_rate"]
            state.pending_order_id = None
            state.maker_volume = 0
            state.volatility_contribution = 0.0
            state.is_liquidated = False
            state.order_counter = 0
            state.maker_fee_rate = cfg["maker_fee_rate"]
            state.taker_fee_rate = cfg["taker_fee_rate"]
            state.bid_order_ids = []
            state.ask_order_ids = []
            state.cumulative_spread_score = 0.0
            state.quote_tick_count = 0

    # 重置噪声交易者状态
    for noise_trader_state in arena.noise_trader_states.values():
        noise_trader_state.reset()

    # 初始化扁平化数组
    arena.init_flat_arrays()


# ============================================================================
# MM 初始化
# ============================================================================


def _execute_mm_init_orders(
    arena: ArenaState,
    agent_state: AgentAccountState,
    bid_orders: list[dict[str, float]],
    ask_orders: list[dict[str, float]],
) -> list[Trade]:
    """执行做市商初始化订单

    Args:
        arena: 竞技场状态
        agent_state: 做市商 Agent 账户状态
        bid_orders: 买单列表
        ask_orders: 卖单列表

    Returns:
        成交列表
    """
    matching_engine = arena.matching_engine
    all_trades: list[Trade] = []

    # 挂买单
    for order_spec in bid_orders:
        order_id = agent_state.generate_order_id(arena.arena_id)
        order = Order(
            order_id=order_id,
            agent_id=agent_state.agent_id,
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            price=order_spec["price"],
            quantity=int(order_spec["quantity"]),
        )
        trades = matching_engine.process_order(order)
        for trade in trades:
            all_trades.append(trade)
            arena.recent_trades.append(trade)
            _update_trade_participants(arena, trade)
        if matching_engine._orderbook.order_map.get(order_id):
            agent_state.bid_order_ids.append(order_id)

    # 挂卖单
    for order_spec in ask_orders:
        order_id = agent_state.generate_order_id(arena.arena_id)
        order = Order(
            order_id=order_id,
            agent_id=agent_state.agent_id,
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            price=order_spec["price"],
            quantity=int(order_spec["quantity"]),
        )
        trades = matching_engine.process_order(order)
        for trade in trades:
            all_trades.append(trade)
            arena.recent_trades.append(trade)
            _update_trade_participants(arena, trade)
        if matching_engine._orderbook.order_map.get(order_id):
            agent_state.ask_order_ids.append(order_id)

    return all_trades


def _init_mm_all_arenas(
    arena_states: list[ArenaState],
    caches: dict[AgentType, Any],
    type_groups: dict[AgentType, tuple[list[int], NDArray[np.int32]]],
    config: Config,
    buffers_list: list[dict[str, NDArray[np.float32]]],
    ema_alpha: float,
) -> None:
    """做市商初始化挂单 - 所有竞技场复用同一推理结果

    所有竞技场在 episode 开始时状态完全相同，因此只需对一个竞技场
    进行一次批量推理，然后将结果复用到所有竞技场。

    Args:
        arena_states: 竞技场状态列表
        caches: BatchNetworkCache 字典
        type_groups: Agent 类型分组
        config: 全局配置
        buffers_list: 市场状态缓冲区列表
        ema_alpha: EMA 平滑系数
    """
    mm_cache = caches.get(AgentType.MARKET_MAKER)
    if mm_cache is None or not mm_cache.is_valid():
        return

    mm_ids, mm_net_indices = type_groups.get(
        AgentType.MARKET_MAKER, ([], np.array([], dtype=np.int32))
    )
    if not mm_ids:
        return

    arena0 = arena_states[0]

    # 用第一个竞技场计算市场状态
    market_state = compute_market_state(arena0, buffers_list[0], ema_alpha)
    tick_size = market_state.tick_size if market_state.tick_size > 0 else 0.01

    # 收集 MM 状态
    mm_states: list[AgentAccountState] = [
        arena0.agent_states[aid] for aid in mm_ids if aid in arena0.agent_states
    ]
    mm_indices_list: list[int] = [
        int(idx) for idx in mm_net_indices[: len(mm_states)]
    ]

    # 批量推理
    try:
        raw = mm_cache.decide_multi_arena_direct(
            [mm_states],
            [market_state],
            [mm_indices_list],
            return_array=True,
        )
        mm_array: NDArray[np.float64] | None = raw.get(0)
    except Exception:
        return

    if mm_array is None or len(mm_array) == 0:
        return

    # 解析订单（参考 _prepare_mm_init_orders 中的解析逻辑）
    mm_orders: list[
        tuple[int, list[dict[str, float]], list[dict[str, float]]]
    ] = []
    for i, agent_id in enumerate(mm_ids):
        if i >= len(mm_array):
            break
        num_bid = int(mm_array[i, 0])
        num_ask = int(mm_array[i, 1])
        bid_orders: list[dict[str, float]] = []
        for j in range(num_bid):
            price = float(mm_array[i, 2 + j])
            qty = float(mm_array[i, 12 + j])
            if qty > 0:
                bid_orders.append({"price": price, "quantity": qty})
        ask_orders: list[dict[str, float]] = []
        for j in range(num_ask):
            price = float(mm_array[i, 22 + j])
            qty = float(mm_array[i, 32 + j])
            if qty > 0:
                ask_orders.append({"price": price, "quantity": qty})
        mm_orders.append((agent_id, bid_orders, ask_orders))

    # 在每个竞技场执行
    for arena in arena_states:
        for agent_id, bid_orders, ask_orders in mm_orders:
            agent_state = arena.agent_states.get(agent_id)
            if agent_state is None:
                continue
            _execute_mm_init_orders(arena, agent_state, bid_orders, ask_orders)

        # 更新 smooth_mid_price
        mid = arena.matching_engine._orderbook.get_mid_price()
        if mid is not None and mid > 0:
            arena.smooth_mid_price = mid


# ============================================================================
# Spread Score 计算
# ============================================================================


def _compute_mm_spread_scores(
    arena: ArenaState,
    mm_states: list[AgentAccountState],
    mm_array: NDArray[np.float64],
    tick_size: float,
) -> None:
    """计算做市商 spread score 并累积到 AgentAccountState

    Args:
        arena: 竞技场状态
        mm_states: 做市商状态列表
        mm_array: 做市商决策数组
        tick_size: 最小价格变动单位
    """
    for i, state in enumerate(mm_states):
        if state.is_liquidated:
            continue
        if i >= len(mm_array):
            break
        num_bid = int(mm_array[i, 0])
        num_ask = int(mm_array[i, 1])
        if num_bid == 0 or num_ask == 0:
            continue
        bid_prices = mm_array[i, 2: 2 + num_bid]
        bid_qtys = mm_array[i, 12: 12 + num_bid]
        ask_prices = mm_array[i, 22: 22 + num_ask]
        ask_qtys = mm_array[i, 32: 32 + num_ask]
        valid_bids = bid_prices[bid_qtys > 0]
        valid_asks = ask_prices[ask_qtys > 0]
        if len(valid_bids) == 0 or len(valid_asks) == 0:
            continue
        best_bid = float(np.max(valid_bids))
        best_ask = float(np.min(valid_asks))
        if best_ask <= best_bid:
            tick_score = 1.0  # 交叉报价，视为最紧盘口
        else:
            quoted_spread = best_ask - best_bid
            max_spread = 200.0 * tick_size
            tick_score = max(0.0, 1.0 - quoted_spread / max_spread)
        state.cumulative_spread_score += tick_score
        state.quote_tick_count += 1


# ============================================================================
# 适应度收集
# ============================================================================


def _collect_fitness_all_arenas(
    worker_id: int,
    arena_states: list[ArenaState],
    agent_infos_by_sub_pop: dict[tuple[AgentType, int], list[AgentInfo]],
    mm_fitness_weights: tuple[float, float, float, float],
    config: Config,
) -> EpisodeResult:
    """收集所有竞技场的适应度

    Args:
        worker_id: Worker ID
        arena_states: 竞技场状态列表
        agent_infos_by_sub_pop: 按 (agent_type, sub_pop_id) 分组的 agent 信息
        mm_fitness_weights: 做市商适应度权重 (alpha, beta, gamma, delta)
        config: 全局配置

    Returns:
        EpisodeResult 汇总结果
    """
    accumulated: dict[tuple[AgentType, int], NDArray[np.float32]] = {}
    per_arena_fitness: dict[int, dict[AgentType, NDArray[np.float32]]] = {}
    arena_stats: dict[int, ArenaEpisodeStats] = {}

    for arena in arena_states:
        current_price = arena.matching_engine._orderbook.last_price
        if current_price <= 0:
            mid = arena.matching_engine._orderbook.get_mid_price()
            current_price = (
                mid if mid is not None and mid > 0 else arena.smooth_mid_price
            )

        arena_fitness_by_type: dict[AgentType, NDArray[np.float32]] = {}

        for key, infos in agent_infos_by_sub_pop.items():
            agent_type, sub_pop_id = key
            fitness_arr = _calculate_fitness_for_sub_pop(
                arena,
                infos,
                current_price,
                mm_fitness_weights,
            )

            # 累积到汇总
            if key not in accumulated:
                accumulated[key] = fitness_arr.copy()
            else:
                accumulated[key] += fitness_arr

            # 按 AgentType 汇总（供 league trainer 使用）
            if agent_type not in arena_fitness_by_type:
                arena_fitness_by_type[agent_type] = fitness_arr.copy()
            else:
                arena_fitness_by_type[agent_type] = np.concatenate(
                    [arena_fitness_by_type[agent_type], fitness_arr]
                )

        per_arena_fitness[arena.arena_id] = arena_fitness_by_type

        arena_stats[arena.arena_id] = ArenaEpisodeStats(
            end_reason=arena.end_reason,
            end_tick=arena.end_tick if arena.end_reason else arena.tick,
            high_price=arena.episode_high_price,
            low_price=arena.episode_low_price,
        )

    return EpisodeResult(
        worker_id=worker_id,
        accumulated_fitness=accumulated,
        per_arena_fitness=per_arena_fitness,
        arena_stats=arena_stats,
    )


def _calculate_fitness_for_sub_pop(
    arena: ArenaState,
    infos: list[AgentInfo],
    current_price: float,
    mm_fitness_weights: tuple[float, float, float, float],
) -> NDArray[np.float32]:
    """计算单个子种群的适应度

    Args:
        arena: 竞技场状态
        infos: 子种群的 Agent 信息列表
        current_price: 当前价格
        mm_fitness_weights: 做市商适应度权重 (alpha, beta, gamma, delta)

    Returns:
        适应度数组
    """
    n = len(infos)
    if n == 0:
        return np.zeros(0, dtype=np.float32)

    agent_type = infos[0].agent_type

    if agent_type == AgentType.MARKET_MAKER:
        # 四组件复合适应度
        pnl_arr = np.zeros(n, dtype=np.float32)
        spread_arr = np.zeros(n, dtype=np.float32)
        volume_arr = np.zeros(n, dtype=np.float32)
        survival_arr = np.zeros(n, dtype=np.float32)

        for idx, info in enumerate(infos):
            state = arena.agent_states.get(info.agent_id)
            if state is None:
                continue
            equity = state.get_equity(current_price)
            initial = state.initial_balance
            if initial > 0:
                pnl_arr[idx] = (equity - initial) / initial
            if state.quote_tick_count > 0:
                spread_arr[idx] = (
                    state.cumulative_spread_score / state.quote_tick_count
                )
            volume_arr[idx] = float(state.maker_volume)
            survival_arr[idx] = 0.0 if state.is_liquidated else 1.0

        max_volume = float(np.max(volume_arr)) if len(volume_arr) > 0 else 0.0
        norm_volume = volume_arr / (max_volume + 1.0)

        alpha, beta, gamma, delta = mm_fitness_weights
        return (
            alpha * pnl_arr
            + beta * spread_arr
            + gamma * norm_volume
            + delta * survival_arr
        )
    else:
        # 非 MM：纯收益率
        fitnesses = np.zeros(n, dtype=np.float32)
        for idx, info in enumerate(infos):
            state = arena.agent_states.get(info.agent_id)
            if state is None:
                continue
            equity = state.get_equity(current_price)
            initial = state.initial_balance
            if initial > 0:
                fitnesses[idx] = (equity - initial) / initial
        return fitnesses


# ============================================================================
# Episode 结果合并
# ============================================================================


def _merge_episode_results(
    worker_id: int, results: list[EpisodeResult]
) -> EpisodeResult:
    """合并多个 episode 的结果

    适应度使用累加（后续由主进程除以总 episode 数取平均）。

    Args:
        worker_id: Worker ID
        results: 各 episode 的结果

    Returns:
        合并后的 EpisodeResult
    """
    if len(results) == 1:
        return results[0]

    merged_accumulated: dict[tuple[AgentType, int], NDArray[np.float32]] = {}
    merged_per_arena: dict[int, dict[AgentType, NDArray[np.float32]]] = {}
    merged_stats: dict[int, ArenaEpisodeStats] = {}

    for result in results:
        for key, fitness_arr in result.accumulated_fitness.items():
            if key not in merged_accumulated:
                merged_accumulated[key] = fitness_arr.copy()
            else:
                merged_accumulated[key] += fitness_arr

        # per_arena_fitness: 取最后一个 episode 的（或累加）
        for arena_id, fitness_by_type in result.per_arena_fitness.items():
            if arena_id not in merged_per_arena:
                merged_per_arena[arena_id] = {}
            for agent_type, fitness_arr in fitness_by_type.items():
                if agent_type not in merged_per_arena[arena_id]:
                    merged_per_arena[arena_id][agent_type] = fitness_arr.copy()
                else:
                    merged_per_arena[arena_id][agent_type] += fitness_arr

        # arena_stats: 取最后一个 episode 的
        for arena_id, stats in result.arena_stats.items():
            merged_stats[arena_id] = stats

    return EpisodeResult(
        worker_id=worker_id,
        accumulated_fitness=merged_accumulated,
        per_arena_fitness=merged_per_arena,
        arena_stats=merged_stats,
    )


# ============================================================================
# 核心：Worker 本地运行一个 Episode
# ============================================================================


def _run_episode_local(
    worker_id: int,
    arena_states: list[ArenaState],
    caches: dict[AgentType, Any],
    type_groups: dict[AgentType, tuple[list[int], NDArray[np.int32]]],
    buffers_list: list[dict[str, NDArray[np.float32]]],
    config: Config,
    episode_length: int,
    pop_total_counts: dict[AgentType, int],
    mm_fitness_weights: tuple[float, float, float, float],
    agent_infos_by_sub_pop: dict[tuple[AgentType, int], list[AgentInfo]],
    agent_infos: list[AgentInfo],
    ema_alpha: float,
    noise_trader_config: NoiseTraderConfig,
) -> EpisodeResult:
    """Worker 独立运行一个 episode 的完整逻辑

    Args:
        worker_id: Worker ID
        arena_states: 竞技场状态列表
        caches: BatchNetworkCache 字典
        type_groups: Agent 类型分组
        buffers_list: 市场状态缓冲区列表
        config: 全局配置
        episode_length: Episode 长度（tick 数）
        pop_total_counts: 每种类型的总 Agent 数量
        mm_fitness_weights: 做市商适应度权重
        agent_infos_by_sub_pop: 按子种群分组的 Agent 信息
        agent_infos: Agent 信息列表
        ema_alpha: EMA 平滑系数
        noise_trader_config: 噪声交易者配置

    Returns:
        EpisodeResult 结果
    """
    # 1. 重置所有竞技场
    for arena in arena_states:
        _reset_arena(arena, config, type_groups, agent_infos)

    # 2. MM 初始化
    _init_mm_all_arenas(
        arena_states, caches, type_groups, config, buffers_list, ema_alpha
    )

    # 3. Tick 循环
    for tick_num in range(episode_length):
        all_ended = True

        for arena_idx, arena in enumerate(arena_states):
            if arena.end_reason is not None:
                continue
            all_ended = False

            arena.tick += 1

            # Tick 1: 只记录 MM 初始化后的状态
            if arena.tick == 1:
                current_price = arena.smooth_mid_price
                arena.price_history.append(current_price)
                arena.tick_history_prices.append(current_price)
                arena.tick_history_volumes.append(0.0)
                arena.tick_history_amounts.append(0.0)
                continue

            # 获取当前价格
            current_price = arena.smooth_mid_price
            if current_price <= 0:
                current_price = arena.matching_engine._orderbook.last_price

            # 强平检查（三阶段，本地执行）
            handle_liquidations(arena, current_price)

            # 噪声交易者决策
            noise_decisions = compute_noise_trader_decisions(
                arena, noise_trader_config
            )

            # 计算市场状态
            market_state = compute_market_state(
                arena, buffers_list[arena_idx], ema_alpha
            )

            # 收集活跃 agent（按类型分组）
            arena_retail_states: list[AgentAccountState] = []
            arena_retail_indices: list[int] = []
            arena_mm_states: list[AgentAccountState] = []
            arena_mm_indices: list[int] = []

            for agent_type, (all_ids, all_net_indices) in type_groups.items():
                for j, agent_id in enumerate(all_ids):
                    state = arena.agent_states.get(agent_id)
                    if state is None or state.is_liquidated:
                        continue
                    if agent_type == AgentType.MARKET_MAKER:
                        arena_mm_states.append(state)
                        arena_mm_indices.append(int(all_net_indices[j]))
                    else:
                        arena_retail_states.append(state)
                        arena_retail_indices.append(int(all_net_indices[j]))

            # 打乱非 MM 的顺序
            combined = list(zip(arena_retail_states, arena_retail_indices))
            random.shuffle(combined)
            if combined:
                arena_retail_states_shuffled, arena_retail_indices_shuffled = zip(
                    *combined
                )
                arena_retail_states = list(arena_retail_states_shuffled)
                arena_retail_indices = list(arena_retail_indices_shuffled)

            # 批量推理
            retail_decisions_arr: NDArray[np.float64] | None = None
            retail_agent_ids_arr: NDArray[np.int64] | None = None
            mm_decisions_arr: NDArray[np.float64] | None = None
            mm_agent_ids_arr: NDArray[np.int64] | None = None

            # 推理非 MM
            retail_cache = caches.get(AgentType.RETAIL_PRO)
            if (
                retail_cache is not None
                and retail_cache.is_valid()
                and arena_retail_states
            ):
                try:
                    raw = retail_cache.decide_multi_arena_direct(
                        [arena_retail_states],
                        [market_state],
                        [arena_retail_indices],
                        return_array=True,
                    )
                    result_arr = raw.get(0)
                    if result_arr is not None and len(result_arr) > 0:
                        # 过滤 HOLD 动作
                        non_hold_mask = result_arr[:, 0] != 0
                        non_hold_indices_np = np.where(non_hold_mask)[0]
                        retail_agent_ids_arr = np.array(
                            [
                                arena_retail_states[int(i)].agent_id
                                for i in non_hold_indices_np
                            ],
                            dtype=np.int64,
                        )
                        retail_decisions_arr = result_arr[non_hold_mask]
                except Exception:
                    pass

            # 推理 MM
            mm_cache = caches.get(AgentType.MARKET_MAKER)
            if (
                mm_cache is not None
                and mm_cache.is_valid()
                and arena_mm_states
            ):
                try:
                    raw = mm_cache.decide_multi_arena_direct(
                        [arena_mm_states],
                        [market_state],
                        [arena_mm_indices],
                        return_array=True,
                    )
                    mm_arr = raw.get(0)
                    if mm_arr is not None and len(mm_arr) > 0:
                        mm_agent_ids_arr = np.array(
                            [
                                arena_mm_states[i].agent_id
                                for i in range(len(mm_arr))
                            ],
                            dtype=np.int64,
                        )
                        mm_decisions_arr = mm_arr

                        # 计算 spread score
                        tick_size = (
                            market_state.tick_size
                            if market_state.tick_size > 0
                            else 0.01
                        )
                        _compute_mm_spread_scores(
                            arena, arena_mm_states, mm_arr, tick_size
                        )
                except Exception:
                    pass

            # 本地执行（原子动作模式）
            tick_trades = execute_tick_local(
                arena=arena,
                retail_decisions=retail_decisions_arr,
                retail_agent_ids=retail_agent_ids_arr,
                mm_decisions=mm_decisions_arr,
                mm_agent_ids=mm_agent_ids_arr,
                noise_decisions=noise_decisions,
                liquidated_agents=[],  # 强平已在上面处理
            )

            # 记录价格历史
            orderbook = arena.matching_engine._orderbook
            actual_price = (
                orderbook.last_price
                if orderbook.last_price > 0
                else arena.smooth_mid_price
            )
            current_price = arena.smooth_mid_price
            arena.price_history.append(current_price)
            update_episode_price_stats_from_trades(
                arena, tick_trades, actual_price
            )

            # 记录 tick 历史
            arena.tick_history_prices.append(current_price)
            volume, amount = aggregate_tick_trades(tick_trades)
            arena.tick_history_volumes.append(volume)
            arena.tick_history_amounts.append(amount)

            # 检查提前结束
            early_end = check_early_end(arena, pop_total_counts)
            if early_end is not None:
                reason, agent_type = early_end
                if agent_type is not None:
                    arena.end_reason = f"{reason}:{agent_type.name}"
                else:
                    arena.end_reason = reason
                arena.end_tick = arena.tick

        if all_ended:
            break

    # 4. 收集适应度
    return _collect_fitness_all_arenas(
        worker_id,
        arena_states,
        agent_infos_by_sub_pop,
        mm_fitness_weights,
        config,
    )


# ============================================================================
# Arena Worker 进程主函数
# ============================================================================


def arena_worker_main(
    worker_id: int,
    arena_ids: list[int],
    config: Config,
    agent_infos: list[AgentInfo],
    cmd_queue: Queue,  # type: ignore[type-arg]
    result_queue: Queue,  # type: ignore[type-arg]
) -> None:
    """Arena Worker 进程主函数

    每个 Worker 持有独立的 BatchNetworkCache、ArenaState、MatchingEngine。
    主循环处理命令：update_networks / run_episode / shutdown。

    Args:
        worker_id: Worker ID
        arena_ids: 本 Worker 负责的竞技场 ID 列表
        config: 全局配置
        agent_infos: Agent 信息列表
        cmd_queue: 命令接收队列
        result_queue: 结果发送队列
    """
    # 设置环境
    os.environ["OMP_NUM_THREADS"] = "1"

    worker_logger = logging.getLogger(f"ArenaWorker-{worker_id}")
    worker_logger.info(
        f"Worker {worker_id} 启动，负责竞技场: {arena_ids}"
    )

    # 创建本地 BatchNetworkCache（每个 Worker 独立）
    caches = _create_worker_caches(config, agent_infos)

    # 创建本地 ArenaState（每个 Worker 的每个竞技场）
    arena_states = _create_worker_arena_states(arena_ids, config, agent_infos)

    # 构建 agent 分组缓存
    type_groups = _build_worker_type_groups(agent_infos)

    # 预分配市场状态缓冲区（每个竞技场一组）
    buffers_list = [_create_market_state_buffers() for _ in arena_ids]

    # 种群总数（用于早期结束检查）
    pop_total_counts = _compute_pop_total_counts(agent_infos)

    # MM 适应度权重
    mm_fitness_weights = (
        config.training.mm_fitness_pnl_weight,
        config.training.mm_fitness_spread_weight,
        config.training.mm_fitness_volume_weight,
        config.training.mm_fitness_survival_weight,
    )

    # 按 (agent_type, sub_pop_id) 分组的 agent_infos
    agent_infos_by_sub_pop = _group_agent_infos_by_sub_pop(agent_infos)

    # EMA 参数
    ema_alpha = config.market.ema_alpha

    # 噪声交易者配置
    noise_trader_config = config.noise_trader

    worker_logger.info(f"Worker {worker_id} 初始化完成")

    try:
        while True:
            cmd_type, cmd_data = cmd_queue.get()

            if cmd_type == "update_networks":
                for agent_type, params in cmd_data.items():
                    if agent_type in caches and caches[agent_type] is not None:
                        caches[agent_type].update_networks_from_numpy(*params)
                result_queue.put((worker_id, "ack"))

            elif cmd_type == "run_episode":
                episode_length: int = cmd_data["episode_length"]
                num_episodes: int = cmd_data["num_episodes"]

                all_results: list[EpisodeResult] = []
                for _ in range(num_episodes):
                    result = _run_episode_local(
                        worker_id=worker_id,
                        arena_states=arena_states,
                        caches=caches,
                        type_groups=type_groups,
                        buffers_list=buffers_list,
                        config=config,
                        episode_length=episode_length,
                        pop_total_counts=pop_total_counts,
                        mm_fitness_weights=mm_fitness_weights,
                        agent_infos_by_sub_pop=agent_infos_by_sub_pop,
                        agent_infos=agent_infos,
                        ema_alpha=ema_alpha,
                        noise_trader_config=noise_trader_config,
                    )
                    all_results.append(result)

                # 如果多个 episode，合并结果
                if len(all_results) == 1:
                    result_queue.put((worker_id, all_results[0]))
                else:
                    merged = _merge_episode_results(worker_id, all_results)
                    result_queue.put((worker_id, merged))

            elif cmd_type == "shutdown":
                worker_logger.info(f"Worker {worker_id} 收到关闭命令")
                break

    except KeyboardInterrupt:
        worker_logger.info(f"Worker {worker_id} 被中断")
    except Exception as e:
        worker_logger.error(f"Worker {worker_id} 发生异常: {e}")
    finally:
        # 清理
        for cache in caches.values():
            if cache is not None:
                try:
                    cache.clear()
                except Exception:
                    pass
        worker_logger.info(f"Worker {worker_id} 退出")


# ============================================================================
# ArenaWorkerPool 管理类
# ============================================================================


class ArenaWorkerPool:
    """管理持久化 Arena Worker 进程池

    主进程通过此类与 Worker 进程通信：
    - update_networks(): 进化后发送新网络参数
    - run_episodes(): 所有 Worker 独立运行 episodes 并返回适应度结果
    - shutdown(): 关闭所有 Worker

    Attributes:
        _num_workers: Worker 进程数量
        _num_arenas: 竞技场总数
        _config: 全局配置
        _agent_infos: Agent 信息列表
        _worker_arena_ids: 每个 Worker 分配的竞技场 ID 列表
        _cmd_queues: 每个 Worker 的命令队列
        _result_queue: 统一结果队列
        _workers: Worker 进程列表
        _is_started: 是否已启动
    """

    def __init__(
        self,
        num_workers: int,
        num_arenas: int,
        config: Config,
        agent_infos: list[AgentInfo],
    ) -> None:
        """初始化 Worker 池

        Args:
            num_workers: Worker 进程数量
            num_arenas: 竞技场总数
            config: 全局配置
            agent_infos: Agent 信息列表
        """
        self._num_workers = num_workers
        self._num_arenas = num_arenas
        self._config = config
        self._agent_infos = agent_infos

        # 每个 Worker 分配的竞技场 ID
        self._worker_arena_ids: list[list[int]] = []
        arenas_per_worker = num_arenas // num_workers
        remainder = num_arenas % num_workers
        idx = 0
        for w in range(num_workers):
            count = arenas_per_worker + (1 if w < remainder else 0)
            self._worker_arena_ids.append(list(range(idx, idx + count)))
            idx += count

        # IPC 队列
        self._cmd_queues: list[Queue] = []  # type: ignore[type-arg]
        self._result_queue: Queue = Queue()  # type: ignore[type-arg]
        self._workers: list[Process] = []
        self._is_started = False
        self._logger = logging.getLogger("ArenaWorkerPool")

    def start(self) -> None:
        """启动所有 Worker 进程"""
        if self._is_started:
            return

        for w in range(self._num_workers):
            cmd_queue: Queue = Queue()  # type: ignore[type-arg]
            self._cmd_queues.append(cmd_queue)

            p = Process(
                target=arena_worker_main,
                args=(
                    w,
                    self._worker_arena_ids[w],
                    self._config,
                    self._agent_infos,
                    cmd_queue,
                    self._result_queue,
                ),
                daemon=True,
            )
            p.start()
            self._workers.append(p)

        self._is_started = True
        self._logger.info(
            f"ArenaWorkerPool 已启动: {self._num_workers} 个 Worker, "
            f"{self._num_arenas} 个竞技场"
        )

    def update_networks(
        self,
        network_params: dict[AgentType, tuple[np.ndarray, ...]],
    ) -> None:
        """进化后发送新网络参数给所有 Worker

        Args:
            network_params: 每种类型的网络参数元组
        """
        if not self._is_started:
            self.start()

        for cmd_queue in self._cmd_queues:
            cmd_queue.put(("update_networks", network_params))
        # 等待所有 Worker 确认
        for _ in range(self._num_workers):
            self._result_queue.get()

    def run_episodes(
        self,
        num_episodes: int,
        episode_length: int,
    ) -> list[EpisodeResult]:
        """所有 Worker 独立运行 episodes，返回汇总结果

        Args:
            num_episodes: 每个 Worker 运行的 episode 数量
            episode_length: 每个 episode 的 tick 数

        Returns:
            各 Worker 的 EpisodeResult 列表
        """
        if not self._is_started:
            self.start()

        for cmd_queue in self._cmd_queues:
            cmd_queue.put(
                (
                    "run_episode",
                    {
                        "num_episodes": num_episodes,
                        "episode_length": episode_length,
                    },
                )
            )

        results: list[EpisodeResult] = []
        for _ in range(self._num_workers):
            _worker_id, result = self._result_queue.get()
            results.append(result)

        return results

    def shutdown(self) -> None:
        """关闭所有 Worker"""
        if not self._is_started:
            return

        self._logger.info("正在关闭 ArenaWorkerPool...")

        for cmd_queue in self._cmd_queues:
            cmd_queue.put(("shutdown", None))
        for p in self._workers:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()
        self._workers.clear()
        self._cmd_queues.clear()
        self._is_started = False
        self._logger.info("ArenaWorkerPool 已关闭")

    def __enter__(self) -> "ArenaWorkerPool":
        """上下文管理器入口"""
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """上下文管理器退出"""
        self.shutdown()
