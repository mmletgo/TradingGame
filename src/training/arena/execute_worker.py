"""Execute Worker Pool 模块

本模块实现竞技场执行的 Worker 池，将 Execute 阶段并行化。
每个 Worker 负责若干个竞技场的订单执行，维护独立的 OrderBook 和 MatchingEngine。

主要组件:
- ExecuteCommand: 主进程 -> Worker 的命令数据类
- ArenaExecuteData: execute 命令的数据
- ArenaExecuteResult: Worker -> 主进程的结果数据类
- ArenaExecuteWorkerPool: 管理多个 Worker 进程的池（Queue 版）
- arena_execute_worker: Worker 进程主函数（Queue 版）
- ArenaExecuteWorkerPoolShm: 共享内存版 Worker 池
- arena_execute_worker_shm: 共享内存版 Worker 进程主函数
"""

from __future__ import annotations

import logging
import traceback
from collections import deque
from dataclasses import dataclass, field
from multiprocessing import Process, Queue
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from src.config.config import AgentType, Config
from src.market.matching.matching_engine import MatchingEngine
from src.market.matching.trade import Trade
from src.market.orderbook.order import Order, OrderSide, OrderType

if TYPE_CHECKING:
    from multiprocessing import Queue as QueueType
    from multiprocessing.synchronize import Event


# ============================================================================
# 数据类定义
# ============================================================================


@dataclass
class ExecuteCommand:
    """主进程 -> Worker 的命令

    Attributes:
        cmd_type: 命令类型，支持 "reset", "init_mm", "execute", "get_depth", "shutdown"
        arena_id: 竞技场 ID
        data: 命令特定数据
    """

    cmd_type: str  # "reset", "init_mm", "execute", "get_depth", "shutdown"
    arena_id: int
    data: Any = None


@dataclass
class ArenaExecuteData:
    """execute 命令的数据

    Attributes:
        liquidated_agents: 需要强平的 Agent 列表，
            格式: [(agent_id, position_qty, is_mm), ...]
        decisions: 非做市商决策列表，
            格式: [(agent_id, action_int, side_int, price, quantity), ...]
        mm_decisions: 做市商决策列表，
            格式: [(agent_id, bid_orders, ask_orders), ...]
            其中 bid_orders/ask_orders 格式: [{"price": float, "quantity": float}, ...]
    """

    liquidated_agents: list[tuple[int, int, bool]] = field(default_factory=list)
    decisions: list[tuple[int, int, int, float, int]] = field(default_factory=list)
    mm_decisions: list[tuple[int, list[dict[str, float]], list[dict[str, float]]]] = (
        field(default_factory=list)
    )


@dataclass
class ArenaExecuteResult:
    """Worker -> 主进程的结果

    Attributes:
        arena_id: 竞技场 ID
        bid_depth: 买盘深度数据，shape (100, 2) - (price, quantity)
        ask_depth: 卖盘深度数据，shape (100, 2) - (price, quantity)
        last_price: 最新成交价
        mid_price: 中间价
        trades: 成交列表，每个元素格式:
            (trade_id, price, quantity, buyer_id, seller_id,
             buyer_fee, seller_fee, is_buyer_taker)
        pending_updates: 非做市商挂单状态更新，agent_id -> pending_order_id | None
        mm_order_updates: 做市商挂单状态更新，agent_id -> (bid_ids, ask_ids)
        error: 错误信息（如果有）
    """

    arena_id: int
    bid_depth: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros((100, 2), dtype=np.float64)
    )
    ask_depth: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros((100, 2), dtype=np.float64)
    )
    last_price: float = 0.0
    mid_price: float = 0.0
    trades: list[tuple[int, float, int, int, int, float, float, bool]] = field(
        default_factory=list
    )
    pending_updates: dict[int, int | None] = field(default_factory=dict)
    mm_order_updates: dict[int, tuple[list[int], list[int]]] = field(
        default_factory=dict
    )
    error: str | None = None


# ============================================================================
# Worker 状态管理
# ============================================================================


@dataclass
class WorkerArenaState:
    """Worker 维护的单个竞技场状态

    Attributes:
        arena_id: 竞技场 ID
        matching_engine: 撮合引擎实例
        recent_trades: 最近成交记录队列
        pending_order_ids: 非做市商的挂单 ID 映射，agent_id -> order_id
        mm_bid_order_ids: 做市商买单挂单 ID 列表，agent_id -> [order_id, ...]
        mm_ask_order_ids: 做市商卖单挂单 ID 列表，agent_id -> [order_id, ...]
        order_counters: 各 Agent 的订单计数器，agent_id -> counter
    """

    arena_id: int
    matching_engine: MatchingEngine
    recent_trades: deque[Trade] = field(default_factory=lambda: deque(maxlen=100))
    pending_order_ids: dict[int, int | None] = field(default_factory=dict)
    mm_bid_order_ids: dict[int, list[int]] = field(default_factory=dict)
    mm_ask_order_ids: dict[int, list[int]] = field(default_factory=dict)
    order_counters: dict[int, int] = field(default_factory=dict)


# ============================================================================
# 订单 ID 生成
# ============================================================================


def generate_order_id(arena_id: int, agent_id: int, order_counter: int) -> int:
    """生成全局唯一的订单 ID

    订单 ID 结构（64位）:
    - 高 24 位：arena_id
    - 中 24 位：agent_id（最大支持约 1600 万）
    - 低 16 位：order_counter

    Args:
        arena_id: 竞技场 ID
        agent_id: Agent ID
        order_counter: 订单计数器

    Returns:
        全局唯一的订单 ID
    """
    return (arena_id << 40) | (agent_id << 16) | (order_counter & 0xFFFF)


# ============================================================================
# Worker 进程函数
# ============================================================================


def arena_execute_worker(
    worker_id: int,
    arena_ids: list[int],
    config: Config,
    cmd_queue: "QueueType[ExecuteCommand]",
    result_queue: "QueueType[ArenaExecuteResult | tuple[str, int]]",
) -> None:
    """Worker 进程主函数

    每个 Worker 负责若干个竞技场的订单执行。
    维护独立的 MatchingEngine 和 OrderBook。

    Args:
        worker_id: Worker ID（用于日志）
        arena_ids: 本 Worker 负责的竞技场 ID 列表
        config: 全局配置
        cmd_queue: 命令接收队列
        result_queue: 结果发送队列
    """
    logger = logging.getLogger(f"ExecuteWorker-{worker_id}")
    logger.info(f"Worker {worker_id} 启动，负责竞技场: {arena_ids}")

    # 初始化各竞技场状态
    arena_states: dict[int, WorkerArenaState] = {}
    for arena_id in arena_ids:
        matching_engine = MatchingEngine(config.market)
        arena_states[arena_id] = WorkerArenaState(
            arena_id=arena_id,
            matching_engine=matching_engine,
        )

    try:
        while True:
            # 接收命令
            cmd = cmd_queue.get()

            if cmd.cmd_type == "shutdown":
                logger.info(f"Worker {worker_id} 收到关闭命令")
                break

            arena_id = cmd.arena_id
            if arena_id not in arena_states:
                logger.warning(f"未知的竞技场 ID: {arena_id}")
                continue

            arena = arena_states[arena_id]

            try:
                if cmd.cmd_type == "reset":
                    # 重置竞技场订单簿
                    _handle_reset(arena, config, cmd.data)
                    result_queue.put(("reset_done", arena_id))

                elif cmd.cmd_type == "init_mm":
                    # 初始化做市商挂单
                    result = _handle_init_mm(arena, cmd.data)
                    result_queue.put(result)

                elif cmd.cmd_type == "execute":
                    # 执行订单
                    result = _handle_execute(arena, cmd.data)
                    result_queue.put(result)

                elif cmd.cmd_type == "get_depth":
                    # 获取订单簿深度
                    result = _handle_get_depth(arena)
                    result_queue.put(result)

            except Exception as e:
                error_msg = f"Worker {worker_id} 处理命令 {cmd.cmd_type} 失败: {e}\n{traceback.format_exc()}"
                logger.error(error_msg)
                # 返回错误结果
                error_result = ArenaExecuteResult(
                    arena_id=arena_id,
                    error=error_msg,
                )
                result_queue.put(error_result)

    except KeyboardInterrupt:
        logger.info(f"Worker {worker_id} 被中断")
    except Exception as e:
        logger.error(f"Worker {worker_id} 发生异常: {e}\n{traceback.format_exc()}")
    finally:
        logger.info(f"Worker {worker_id} 退出")


def _handle_reset(
    arena: WorkerArenaState,
    config: Config,
    reset_data: dict[str, Any] | None,
) -> None:
    """处理 reset 命令

    重置订单簿和所有状态。

    Args:
        arena: 竞技场状态
        config: 全局配置
        reset_data: 重置数据，可包含 "initial_price" 和 "fee_rates"
    """
    initial_price = config.market.initial_price
    if reset_data and "initial_price" in reset_data:
        initial_price = reset_data["initial_price"]

    # 重置订单簿
    arena.matching_engine._orderbook.clear(reset_price=initial_price)

    # 清空状态
    arena.recent_trades.clear()
    arena.pending_order_ids.clear()
    arena.mm_bid_order_ids.clear()
    arena.mm_ask_order_ids.clear()
    arena.order_counters.clear()

    # 注册 Agent 费率
    if reset_data and "fee_rates" in reset_data:
        for agent_id, (maker_rate, taker_rate) in reset_data["fee_rates"].items():
            arena.matching_engine.register_agent(agent_id, maker_rate, taker_rate)


def _handle_init_mm(
    arena: WorkerArenaState,
    init_data: list[tuple[int, list[dict[str, float]], list[dict[str, float]]]],
) -> ArenaExecuteResult:
    """处理 init_mm 命令（做市商初始化挂单）

    Args:
        arena: 竞技场状态
        init_data: 做市商初始化数据列表，
            格式: [(agent_id, bid_orders, ask_orders), ...]

    Returns:
        执行结果
    """
    all_trades: list[tuple[int, float, int, int, int, float, float, bool]] = []
    mm_order_updates: dict[int, tuple[list[int], list[int]]] = {}

    matching_engine = arena.matching_engine
    orderbook = matching_engine._orderbook
    process_order = matching_engine.process_order
    order_map_get = orderbook.order_map.get

    for agent_id, bid_orders, ask_orders in init_data:
        # 初始化订单计数器
        if agent_id not in arena.order_counters:
            arena.order_counters[agent_id] = 0

        bid_ids: list[int] = []
        ask_ids: list[int] = []

        # 挂买单
        for order_spec in bid_orders:
            arena.order_counters[agent_id] += 1
            order_id = generate_order_id(
                arena.arena_id, agent_id, arena.order_counters[agent_id]
            )
            order = Order(
                order_id=order_id,
                agent_id=agent_id,
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                price=order_spec["price"],
                quantity=int(order_spec["quantity"]),
            )
            trades = process_order(order)
            for trade in trades:
                all_trades.append(_trade_to_tuple(trade))
                arena.recent_trades.append(trade)
            if order_map_get(order_id):
                bid_ids.append(order_id)

        # 挂卖单
        for order_spec in ask_orders:
            arena.order_counters[agent_id] += 1
            order_id = generate_order_id(
                arena.arena_id, agent_id, arena.order_counters[agent_id]
            )
            order = Order(
                order_id=order_id,
                agent_id=agent_id,
                side=OrderSide.SELL,
                order_type=OrderType.LIMIT,
                price=order_spec["price"],
                quantity=int(order_spec["quantity"]),
            )
            trades = process_order(order)
            for trade in trades:
                all_trades.append(_trade_to_tuple(trade))
                arena.recent_trades.append(trade)
            if order_map_get(order_id):
                ask_ids.append(order_id)

        # 记录挂单 ID
        arena.mm_bid_order_ids[agent_id] = bid_ids
        arena.mm_ask_order_ids[agent_id] = ask_ids
        mm_order_updates[agent_id] = (bid_ids, ask_ids)

    # 构建结果
    bid_depth, ask_depth = _get_depth_arrays(orderbook)
    mid_price = orderbook.get_mid_price() or orderbook.last_price

    return ArenaExecuteResult(
        arena_id=arena.arena_id,
        bid_depth=bid_depth,
        ask_depth=ask_depth,
        last_price=orderbook.last_price,
        mid_price=mid_price,
        trades=all_trades,
        pending_updates={},
        mm_order_updates=mm_order_updates,
    )


def _handle_execute(
    arena: WorkerArenaState,
    execute_data: ArenaExecuteData,
) -> ArenaExecuteResult:
    """处理 execute 命令（执行订单）

    执行顺序：
    1. 强平处理（撤单 + 市价平仓）
    2. 做市商执行（撤旧单 -> 挂新单）
    3. 非做市商执行

    Args:
        arena: 竞技场状态
        execute_data: 执行数据

    Returns:
        执行结果
    """
    all_trades: list[tuple[int, float, int, int, int, float, float, bool]] = []
    pending_updates: dict[int, int | None] = {}
    mm_order_updates: dict[int, tuple[list[int], list[int]]] = {}

    matching_engine = arena.matching_engine
    orderbook = matching_engine._orderbook
    process_order = matching_engine.process_order
    cancel_order = matching_engine.cancel_order
    order_map_get = orderbook.order_map.get

    # 1. 处理强平
    for agent_id, position_qty, is_mm in execute_data.liquidated_agents:
        # 撤单
        if is_mm:
            for order_id in arena.mm_bid_order_ids.get(agent_id, []):
                cancel_order(order_id)
            for order_id in arena.mm_ask_order_ids.get(agent_id, []):
                cancel_order(order_id)
            arena.mm_bid_order_ids[agent_id] = []
            arena.mm_ask_order_ids[agent_id] = []
            mm_order_updates[agent_id] = ([], [])
        else:
            pending_id = arena.pending_order_ids.get(agent_id)
            if pending_id is not None:
                cancel_order(pending_id)
            arena.pending_order_ids[agent_id] = None
            pending_updates[agent_id] = None

        # 市价平仓
        if position_qty != 0:
            if agent_id not in arena.order_counters:
                arena.order_counters[agent_id] = 0
            arena.order_counters[agent_id] += 1
            order_id = generate_order_id(
                arena.arena_id, agent_id, arena.order_counters[agent_id]
            )

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
                all_trades.append(_trade_to_tuple(trade))
                arena.recent_trades.append(trade)

    # 2. 处理做市商决策
    for agent_id, bid_orders, ask_orders in execute_data.mm_decisions:
        # 确保订单计数器存在
        if agent_id not in arena.order_counters:
            arena.order_counters[agent_id] = 0

        # 撤销旧挂单
        for order_id in arena.mm_bid_order_ids.get(agent_id, []):
            cancel_order(order_id)
        for order_id in arena.mm_ask_order_ids.get(agent_id, []):
            cancel_order(order_id)

        bid_ids: list[int] = []
        ask_ids: list[int] = []

        # 挂买单
        for order_spec in bid_orders:
            arena.order_counters[agent_id] += 1
            order_id = generate_order_id(
                arena.arena_id, agent_id, arena.order_counters[agent_id]
            )
            order = Order(
                order_id=order_id,
                agent_id=agent_id,
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                price=order_spec["price"],
                quantity=int(order_spec["quantity"]),
            )
            trades = process_order(order)
            for trade in trades:
                all_trades.append(_trade_to_tuple(trade))
                arena.recent_trades.append(trade)
            if order_map_get(order_id):
                bid_ids.append(order_id)

        # 挂卖单
        for order_spec in ask_orders:
            arena.order_counters[agent_id] += 1
            order_id = generate_order_id(
                arena.arena_id, agent_id, arena.order_counters[agent_id]
            )
            order = Order(
                order_id=order_id,
                agent_id=agent_id,
                side=OrderSide.SELL,
                order_type=OrderType.LIMIT,
                price=order_spec["price"],
                quantity=int(order_spec["quantity"]),
            )
            trades = process_order(order)
            for trade in trades:
                all_trades.append(_trade_to_tuple(trade))
                arena.recent_trades.append(trade)
            if order_map_get(order_id):
                ask_ids.append(order_id)

        # 更新挂单记录
        arena.mm_bid_order_ids[agent_id] = bid_ids
        arena.mm_ask_order_ids[agent_id] = ask_ids
        mm_order_updates[agent_id] = (bid_ids, ask_ids)

    # 3. 处理非做市商决策
    for agent_id, action_int, side_int, price, quantity in execute_data.decisions:
        # 确保订单计数器存在
        if agent_id not in arena.order_counters:
            arena.order_counters[agent_id] = 0

        # action_int: 0=HOLD, 1=PLACE_BID, 2=PLACE_ASK, 3=CANCEL, 4=MARKET_BUY, 5=MARKET_SELL
        if action_int == 0:  # HOLD
            continue

        elif action_int in (1, 2):  # PLACE_BID / PLACE_ASK
            # 先撤旧单
            pending_id = arena.pending_order_ids.get(agent_id)
            if pending_id is not None:
                cancel_order(pending_id)
                arena.pending_order_ids[agent_id] = None

            arena.order_counters[agent_id] += 1
            order_id = generate_order_id(
                arena.arena_id, agent_id, arena.order_counters[agent_id]
            )
            side = OrderSide.BUY if action_int == 1 else OrderSide.SELL
            order = Order(
                order_id=order_id,
                agent_id=agent_id,
                side=side,
                order_type=OrderType.LIMIT,
                price=price,
                quantity=quantity,
            )
            trades = process_order(order)
            for trade in trades:
                all_trades.append(_trade_to_tuple(trade))
                arena.recent_trades.append(trade)

            # 记录挂单
            if order_map_get(order_id):
                arena.pending_order_ids[agent_id] = order_id
                pending_updates[agent_id] = order_id
            else:
                arena.pending_order_ids[agent_id] = None
                pending_updates[agent_id] = None

        elif action_int == 3:  # CANCEL
            pending_id = arena.pending_order_ids.get(agent_id)
            if pending_id is not None:
                cancel_order(pending_id)
            arena.pending_order_ids[agent_id] = None
            pending_updates[agent_id] = None

        elif action_int in (4, 5):  # MARKET_BUY / MARKET_SELL
            arena.order_counters[agent_id] += 1
            order_id = generate_order_id(
                arena.arena_id, agent_id, arena.order_counters[agent_id]
            )
            side = OrderSide.BUY if action_int == 4 else OrderSide.SELL
            order = Order(
                order_id=order_id,
                agent_id=agent_id,
                side=side,
                order_type=OrderType.MARKET,
                price=0.0,
                quantity=quantity,
            )
            trades = process_order(order)
            for trade in trades:
                all_trades.append(_trade_to_tuple(trade))
                arena.recent_trades.append(trade)

    # 构建结果
    bid_depth, ask_depth = _get_depth_arrays(orderbook)
    mid_price = orderbook.get_mid_price() or orderbook.last_price

    return ArenaExecuteResult(
        arena_id=arena.arena_id,
        bid_depth=bid_depth,
        ask_depth=ask_depth,
        last_price=orderbook.last_price,
        mid_price=mid_price,
        trades=all_trades,
        pending_updates=pending_updates,
        mm_order_updates=mm_order_updates,
    )


def _handle_get_depth(arena: WorkerArenaState) -> ArenaExecuteResult:
    """处理 get_depth 命令

    Args:
        arena: 竞技场状态

    Returns:
        包含订单簿深度的结果
    """
    orderbook = arena.matching_engine._orderbook
    bid_depth, ask_depth = _get_depth_arrays(orderbook)
    mid_price = orderbook.get_mid_price() or orderbook.last_price

    return ArenaExecuteResult(
        arena_id=arena.arena_id,
        bid_depth=bid_depth,
        ask_depth=ask_depth,
        last_price=orderbook.last_price,
        mid_price=mid_price,
    )


def _trade_to_tuple(
    trade: Trade,
) -> tuple[int, float, int, int, int, float, float, bool]:
    """将 Trade 对象转换为元组

    Args:
        trade: Trade 对象

    Returns:
        (trade_id, price, quantity, buyer_id, seller_id,
         buyer_fee, seller_fee, is_buyer_taker)
    """
    return (
        trade.trade_id,
        trade.price,
        trade.quantity,
        trade.buyer_id,
        trade.seller_id,
        trade.buyer_fee,
        trade.seller_fee,
        trade.is_buyer_taker,
    )


def _get_depth_arrays(orderbook: Any) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """获取订单簿深度数组

    Args:
        orderbook: OrderBook 实例

    Returns:
        (bid_depth, ask_depth)，各 shape (100, 2)
    """
    depth = orderbook.get_depth(levels=100)
    bid_depth = np.zeros((100, 2), dtype=np.float64)
    ask_depth = np.zeros((100, 2), dtype=np.float64)

    for i, (price, qty) in enumerate(depth.get("bids", [])):
        if i >= 100:
            break
        bid_depth[i, 0] = price
        bid_depth[i, 1] = qty

    for i, (price, qty) in enumerate(depth.get("asks", [])):
        if i >= 100:
            break
        ask_depth[i, 0] = price
        ask_depth[i, 1] = qty

    return bid_depth, ask_depth


# ============================================================================
# Worker 池管理
# ============================================================================


class ArenaExecuteWorkerPool:
    """竞技场执行 Worker 池

    管理多个 Worker 进程，每个 Worker 负责若干个竞技场的订单执行。

    Attributes:
        num_workers: Worker 进程数量
        arena_ids: 所有竞技场 ID 列表
        config: 全局配置
        workers: Worker 进程列表
        cmd_queues: 每个 Worker 的命令队列
        result_queue: 统一的结果队列
        arena_to_worker: 竞技场 ID 到 Worker 索引的映射
    """

    def __init__(
        self,
        num_workers: int,
        arena_ids: list[int],
        config: Config,
    ) -> None:
        """初始化 Worker 池

        Args:
            num_workers: Worker 进程数量（建议 4-8）
            arena_ids: 所有竞技场 ID 列表
            config: 全局配置
        """
        self.num_workers = num_workers
        self.arena_ids = arena_ids
        self.config = config
        self.logger = logging.getLogger("ArenaExecuteWorkerPool")

        self.workers: list[Process] = []
        self.cmd_queues: list["QueueType[ExecuteCommand]"] = []
        self.result_queue: "QueueType[ArenaExecuteResult | tuple[str, int]]" = Queue()
        self.arena_to_worker: dict[int, int] = {}

        self._started = False

    def start(self) -> None:
        """启动所有 Worker 进程"""
        if self._started:
            return

        # 将竞技场分配给各 Worker
        arenas_per_worker = len(self.arena_ids) // self.num_workers
        remainder = len(self.arena_ids) % self.num_workers

        idx = 0
        for worker_id in range(self.num_workers):
            # 分配竞技场
            count = arenas_per_worker + (1 if worker_id < remainder else 0)
            worker_arena_ids = self.arena_ids[idx : idx + count]
            idx += count

            # 记录映射
            for arena_id in worker_arena_ids:
                self.arena_to_worker[arena_id] = worker_id

            # 创建命令队列
            cmd_queue: "QueueType[ExecuteCommand]" = Queue()
            self.cmd_queues.append(cmd_queue)

            # 创建并启动 Worker
            worker = Process(
                target=arena_execute_worker,
                args=(
                    worker_id,
                    worker_arena_ids,
                    self.config,
                    cmd_queue,
                    self.result_queue,
                ),
                daemon=True,
            )
            worker.start()
            self.workers.append(worker)

        self._started = True
        self.logger.info(
            f"Worker 池已启动：{self.num_workers} 个 Worker，"
            f"{len(self.arena_ids)} 个竞技场"
        )

    def reset_all(
        self,
        initial_price: float | None = None,
        fee_rates: dict[int, tuple[float, float]] | None = None,
    ) -> None:
        """重置所有竞技场的订单簿

        Args:
            initial_price: 初始价格（可选，默认使用配置）
            fee_rates: Agent 费率映射（可选），agent_id -> (maker_rate, taker_rate)
        """
        if not self._started:
            self.start()

        reset_data: dict[str, Any] = {}
        if initial_price is not None:
            reset_data["initial_price"] = initial_price
        if fee_rates is not None:
            reset_data["fee_rates"] = fee_rates

        # 发送 reset 命令到所有竞技场
        for arena_id in self.arena_ids:
            worker_idx = self.arena_to_worker[arena_id]
            cmd = ExecuteCommand(
                cmd_type="reset",
                arena_id=arena_id,
                data=reset_data if reset_data else None,
            )
            self.cmd_queues[worker_idx].put(cmd)

        # 等待所有 reset 完成
        completed = 0
        while completed < len(self.arena_ids):
            result = self.result_queue.get()
            if isinstance(result, tuple) and result[0] == "reset_done":
                completed += 1

        self.logger.debug(f"所有 {len(self.arena_ids)} 个竞技场已重置")

    def init_market_makers(
        self,
        mm_init_orders: dict[int, list[tuple[int, list[dict[str, float]], list[dict[str, float]]]]],
    ) -> dict[int, ArenaExecuteResult]:
        """初始化做市商挂单（Episode 开始时调用）

        Args:
            mm_init_orders: 各竞技场的做市商初始化数据
                格式: {arena_id: [(agent_id, bid_orders, ask_orders), ...], ...}

        Returns:
            各竞技场的执行结果
        """
        if not self._started:
            self.start()

        # 发送 init_mm 命令
        for arena_id, init_data in mm_init_orders.items():
            worker_idx = self.arena_to_worker[arena_id]
            cmd = ExecuteCommand(
                cmd_type="init_mm",
                arena_id=arena_id,
                data=init_data,
            )
            self.cmd_queues[worker_idx].put(cmd)

        # 收集结果
        results: dict[int, ArenaExecuteResult] = {}
        while len(results) < len(mm_init_orders):
            result = self.result_queue.get()
            if isinstance(result, ArenaExecuteResult):
                results[result.arena_id] = result

        return results

    def execute_all(
        self,
        arena_commands: dict[int, ArenaExecuteData],
    ) -> dict[int, ArenaExecuteResult]:
        """执行所有竞技场的决策（每个 tick 调用）

        Args:
            arena_commands: 各竞技场的执行数据
                格式: {arena_id: ArenaExecuteData, ...}

        Returns:
            各竞技场的执行结果
        """
        if not self._started:
            self.start()

        # 发送 execute 命令
        for arena_id, execute_data in arena_commands.items():
            worker_idx = self.arena_to_worker[arena_id]
            cmd = ExecuteCommand(
                cmd_type="execute",
                arena_id=arena_id,
                data=execute_data,
            )
            self.cmd_queues[worker_idx].put(cmd)

        # 收集结果
        results: dict[int, ArenaExecuteResult] = {}
        while len(results) < len(arena_commands):
            result = self.result_queue.get()
            if isinstance(result, ArenaExecuteResult):
                results[result.arena_id] = result
                if result.error:
                    self.logger.warning(
                        f"竞技场 {result.arena_id} 执行出错: {result.error}"
                    )

        return results

    def get_all_depths(
        self,
    ) -> dict[int, tuple[NDArray[np.float64], NDArray[np.float64], float, float]]:
        """获取所有竞技场的订单簿深度

        Returns:
            各竞技场的深度数据
            格式: {arena_id: (bid_depth, ask_depth, last_price, mid_price), ...}
        """
        if not self._started:
            self.start()

        # 发送 get_depth 命令
        for arena_id in self.arena_ids:
            worker_idx = self.arena_to_worker[arena_id]
            cmd = ExecuteCommand(
                cmd_type="get_depth",
                arena_id=arena_id,
            )
            self.cmd_queues[worker_idx].put(cmd)

        # 收集结果
        results: dict[int, tuple[NDArray[np.float64], NDArray[np.float64], float, float]] = {}
        while len(results) < len(self.arena_ids):
            result = self.result_queue.get()
            if isinstance(result, ArenaExecuteResult):
                results[result.arena_id] = (
                    result.bid_depth,
                    result.ask_depth,
                    result.last_price,
                    result.mid_price,
                )

        return results

    def shutdown(self) -> None:
        """关闭所有 Worker"""
        if not self._started:
            return

        self.logger.info("正在关闭 Worker 池...")

        # 发送 shutdown 命令
        for worker_idx in range(self.num_workers):
            cmd = ExecuteCommand(
                cmd_type="shutdown",
                arena_id=-1,  # 特殊值表示关闭
            )
            self.cmd_queues[worker_idx].put(cmd)

        # 等待 Worker 退出
        for worker in self.workers:
            worker.join(timeout=5.0)
            if worker.is_alive():
                self.logger.warning(f"Worker {worker.pid} 未能正常退出，强制终止")
                worker.terminate()

        self.workers.clear()
        self.cmd_queues.clear()
        self._started = False
        self.logger.info("Worker 池已关闭")

    def __enter__(self) -> "ArenaExecuteWorkerPool":
        """上下文管理器入口"""
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """上下文管理器退出"""
        self.shutdown()


# ============================================================================
# 共享内存版 Worker 实现
# ============================================================================


def arena_execute_worker_shm(
    worker_id: int,
    arena_ids: list[int],
    config: Config,
    shm_name: str,
    total_arenas: int,
    ready_event: "Event",
) -> None:
    """共享内存版 Worker 进程主函数

    使用 SharedMemoryIPC 进行通信，轮询检查命令状态。

    Args:
        worker_id: Worker ID（用于日志）
        arena_ids: 本 Worker 负责的竞技场 ID 列表
        config: 全局配置
        shm_name: 共享内存名称
        total_arenas: 共享内存中的总竞技场数量
        ready_event: 就绪事件，通知主进程 Worker 已初始化完成
    """
    from src.training.arena.shared_memory_ipc import (
        SharedMemoryIPC,
        CommandStatus,
        CommandType,
    )
    import time

    logger = logging.getLogger(f"ExecuteWorkerShm-{worker_id}")
    logger.info(f"Worker {worker_id} 启动（共享内存模式），负责竞技场: {arena_ids}")

    # 连接共享内存
    ipc = SharedMemoryIPC(num_arenas=total_arenas, shm_name=shm_name, create=False)
    logger.info(f"Worker {worker_id} 已连接共享内存: {shm_name}")

    # 初始化各竞技场状态
    arena_states: dict[int, WorkerArenaState] = {}
    for arena_id in arena_ids:
        matching_engine = MatchingEngine(config.market)
        arena_states[arena_id] = WorkerArenaState(
            arena_id=arena_id,
            matching_engine=matching_engine,
        )

    # 通知主进程已就绪
    ready_event.set()
    logger.info(f"Worker {worker_id} 已就绪")

    # 主循环
    running = True
    poll_interval = 0.0001  # 100us 轮询间隔

    try:
        while running:
            for arena_id in arena_ids:
                cmd_view = ipc.get_command_view(arena_id)
                result_view = ipc.get_result_view(arena_id)

                # 检查是否有待处理的命令
                if cmd_view.status != CommandStatus.PENDING:
                    continue

                # 标记为处理中
                cmd_view.status = CommandStatus.PROCESSING
                cmd_type = cmd_view.cmd_type

                try:
                    if cmd_type == CommandType.SHUTDOWN:
                        logger.info(f"Worker {worker_id} 收到关闭命令")
                        result_view.status = CommandStatus.DONE
                        running = False
                        break

                    arena = arena_states[arena_id]

                    if cmd_type == CommandType.RESET:
                        # 处理 reset 命令
                        # 注意：reset 数据需要通过额外机制传递，这里使用默认配置
                        _handle_reset(arena, config, None)
                        result_view.status = CommandStatus.DONE

                    elif cmd_type == CommandType.INIT_MM:
                        # 处理做市商初始化
                        # 从共享内存读取做市商决策数据
                        mm_decisions = cmd_view.get_mm_decisions()
                        # 转换格式：需要添加 agent_id
                        # 这里假设 mm_decisions 已经包含正确的 agent_id 信息
                        # 实际使用时需要确保数据格式正确
                        result = _handle_init_mm(arena, mm_decisions)
                        _write_result_to_shm(result_view, result)

                    elif cmd_type == CommandType.EXECUTE:
                        # 处理执行命令
                        # 从共享内存读取数据
                        liquidated_agents = cmd_view.get_liquidated()
                        decisions = cmd_view.get_decisions()
                        mm_decisions = cmd_view.get_mm_decisions()

                        # 构建执行数据
                        execute_data = ArenaExecuteData(
                            liquidated_agents=liquidated_agents,
                            decisions=decisions,
                            mm_decisions=mm_decisions,
                        )

                        # 执行
                        result = _handle_execute(arena, execute_data)
                        _write_result_to_shm(result_view, result)

                    elif cmd_type == CommandType.GET_DEPTH:
                        # 处理获取深度命令
                        result = _handle_get_depth(arena)
                        _write_result_to_shm(result_view, result)

                except Exception as e:
                    error_msg = f"Worker {worker_id} 处理命令 {cmd_type} 失败: {e}\n{traceback.format_exc()}"
                    logger.error(error_msg)
                    # 设置完成状态，让主进程知道出错了
                    result_view.status = CommandStatus.DONE

            # 短暂休眠，避免空转消耗 CPU
            if running:
                time.sleep(poll_interval)

    except KeyboardInterrupt:
        logger.info(f"Worker {worker_id} 被中断")
    except Exception as e:
        logger.error(f"Worker {worker_id} 发生异常: {e}\n{traceback.format_exc()}")
    finally:
        # 关闭共享内存连接
        ipc.close()
        logger.info(f"Worker {worker_id} 退出")


def _write_result_to_shm(
    result_view: Any,  # ArenaResultView
    result: ArenaExecuteResult,
) -> None:
    """将执行结果写入共享内存

    Args:
        result_view: 共享内存结果视图
        result: 执行结果
    """
    # 写入深度数据
    result_view.set_depth(result.bid_depth, result.ask_depth)

    # 写入价格
    result_view.set_prices(result.last_price, result.mid_price)

    # 写入成交数据
    result_view.set_trades(result.trades)

    # 写入挂单更新
    result_view.set_pending_updates(result.pending_updates)

    # 写入做市商订单更新
    result_view.set_mm_order_updates(result.mm_order_updates)

    # 标记完成
    result_view.status = 3  # CommandStatus.DONE


class ArenaExecuteWorkerPoolShm:
    """共享内存版竞技场执行 Worker 池

    使用 SharedMemoryIPC 进行通信，避免 Queue 的序列化开销。

    Attributes:
        num_workers: Worker 进程数量
        arena_ids: 所有竞技场 ID 列表
        config: 全局配置
        _ipc: 共享内存 IPC 管理器
        _sync: 共享内存同步器
        _workers: Worker 进程列表
        _ready_events: Worker 就绪事件列表
        _arena_to_worker: 竞技场 ID 到 Worker 索引的映射
        _started: 是否已启动
    """

    def __init__(
        self,
        num_workers: int,
        arena_ids: list[int],
        config: Config,
    ) -> None:
        """初始化共享内存版 Worker 池

        Args:
            num_workers: Worker 进程数量
            arena_ids: 所有竞技场 ID 列表
            config: 全局配置
        """
        from multiprocessing import Event
        from src.training.arena.shared_memory_ipc import (
            SharedMemoryIPC,
            ShmSynchronizer,
        )

        self.num_workers = num_workers
        self.arena_ids = arena_ids
        self.config = config
        self.logger = logging.getLogger("ArenaExecuteWorkerPoolShm")

        # 创建共享内存
        self._ipc = SharedMemoryIPC(num_arenas=len(arena_ids), create=True)
        self._sync = ShmSynchronizer(self._ipc)

        self._workers: list[Process] = []
        self._ready_events: list["Event"] = []
        self._arena_to_worker: dict[int, int] = {}
        self._started = False

    def start(self) -> None:
        """启动所有 Worker 进程"""
        from multiprocessing import Event

        if self._started:
            return

        # 将竞技场分配给各 Worker
        arenas_per_worker = len(self.arena_ids) // self.num_workers
        remainder = len(self.arena_ids) % self.num_workers

        idx = 0
        for worker_id in range(self.num_workers):
            # 分配竞技场
            count = arenas_per_worker + (1 if worker_id < remainder else 0)
            worker_arena_ids = self.arena_ids[idx : idx + count]
            idx += count

            # 记录映射
            for arena_id in worker_arena_ids:
                self._arena_to_worker[arena_id] = worker_id

            # 创建就绪事件
            ready_event: "Event" = Event()
            self._ready_events.append(ready_event)

            # 创建并启动 Worker
            worker = Process(
                target=arena_execute_worker_shm,
                args=(
                    worker_id,
                    worker_arena_ids,
                    self.config,
                    self._ipc.shm_name,
                    len(self.arena_ids),  # total_arenas
                    ready_event,
                ),
                daemon=True,
            )
            worker.start()
            self._workers.append(worker)

        # 等待所有 Worker 就绪
        for i, ready_event in enumerate(self._ready_events):
            if not ready_event.wait(timeout=30.0):
                self.logger.error(f"Worker {i} 启动超时")
                raise RuntimeError(f"Worker {i} 启动超时")

        self._started = True
        self.logger.info(
            f"共享内存版 Worker 池已启动：{self.num_workers} 个 Worker，"
            f"{len(self.arena_ids)} 个竞技场，共享内存: {self._ipc.shm_name}"
        )

    def reset_all(
        self,
        initial_price: float | None = None,
        fee_rates: dict[int, tuple[float, float]] | None = None,
    ) -> None:
        """重置所有竞技场的订单簿

        Args:
            initial_price: 初始价格（可选）
            fee_rates: Agent 费率映射（可选）
        """
        from src.training.arena.shared_memory_ipc import CommandStatus, CommandType

        if not self._started:
            self.start()

        # 发送 reset 命令到所有竞技场
        for arena_id in self.arena_ids:
            cmd_view = self._ipc.get_command_view(arena_id)
            cmd_view.cmd_type = CommandType.RESET
            cmd_view.status = CommandStatus.PENDING

        # 等待所有完成
        if not self._sync.wait_all_done(self.arena_ids, timeout_ms=30000):
            self.logger.error("reset_all 超时")
            raise RuntimeError("reset_all 超时")

        # 重置状态
        self._sync.reset_all_status(self.arena_ids)

        self.logger.debug(f"所有 {len(self.arena_ids)} 个竞技场已重置（共享内存模式）")

    def init_market_makers(
        self,
        mm_init_orders: dict[int, list[tuple[int, list[dict[str, float]], list[dict[str, float]]]]],
    ) -> dict[int, ArenaExecuteResult]:
        """初始化做市商挂单

        Args:
            mm_init_orders: 各竞技场的做市商初始化数据

        Returns:
            各竞技场的执行结果
        """
        from src.training.arena.shared_memory_ipc import CommandStatus, CommandType

        if not self._started:
            self.start()

        # 发送 init_mm 命令
        for arena_id, init_data in mm_init_orders.items():
            cmd_view = self._ipc.get_command_view(arena_id)
            cmd_view.set_mm_decisions(init_data)
            cmd_view.cmd_type = CommandType.INIT_MM
            cmd_view.status = CommandStatus.PENDING

        # 等待所有完成
        arena_ids_list = list(mm_init_orders.keys())
        if not self._sync.wait_all_done(arena_ids_list, timeout_ms=30000):
            self.logger.error("init_market_makers 超时")
            raise RuntimeError("init_market_makers 超时")

        # 读取结果
        results: dict[int, ArenaExecuteResult] = {}
        for arena_id in arena_ids_list:
            result_view = self._ipc.get_result_view(arena_id)
            results[arena_id] = _read_result_from_shm(arena_id, result_view)

        # 重置状态
        self._sync.reset_all_status(arena_ids_list)

        return results

    def execute_all(
        self,
        arena_commands: dict[int, ArenaExecuteData],
    ) -> dict[int, ArenaExecuteResult]:
        """执行所有竞技场的决策

        Args:
            arena_commands: 各竞技场的执行数据

        Returns:
            各竞技场的执行结果
        """
        from src.training.arena.shared_memory_ipc import CommandStatus, CommandType

        if not self._started:
            self.start()

        # 1. 写入命令到共享内存
        for arena_id, execute_data in arena_commands.items():
            cmd_view = self._ipc.get_command_view(arena_id)
            cmd_view.set_liquidated(execute_data.liquidated_agents)
            cmd_view.set_decisions(execute_data.decisions)
            cmd_view.set_mm_decisions(execute_data.mm_decisions)
            cmd_view.cmd_type = CommandType.EXECUTE
            cmd_view.status = CommandStatus.PENDING

        # 2. 等待所有完成
        arena_ids_list = list(arena_commands.keys())
        if not self._sync.wait_all_done(arena_ids_list, timeout_ms=30000):
            self.logger.error("execute_all 超时")
            raise RuntimeError("execute_all 超时")

        # 3. 读取结果
        results: dict[int, ArenaExecuteResult] = {}
        for arena_id in arena_ids_list:
            result_view = self._ipc.get_result_view(arena_id)
            results[arena_id] = _read_result_from_shm(arena_id, result_view)

        # 4. 重置状态
        self._sync.reset_all_status(arena_ids_list)

        return results

    def get_all_depths(
        self,
    ) -> dict[int, tuple[NDArray[np.float64], NDArray[np.float64], float, float]]:
        """获取所有竞技场的订单簿深度

        Returns:
            各竞技场的深度数据
        """
        from src.training.arena.shared_memory_ipc import CommandStatus, CommandType

        if not self._started:
            self.start()

        # 发送 get_depth 命令
        for arena_id in self.arena_ids:
            cmd_view = self._ipc.get_command_view(arena_id)
            cmd_view.cmd_type = CommandType.GET_DEPTH
            cmd_view.status = CommandStatus.PENDING

        # 等待所有完成
        if not self._sync.wait_all_done(self.arena_ids, timeout_ms=30000):
            self.logger.error("get_all_depths 超时")
            raise RuntimeError("get_all_depths 超时")

        # 读取结果
        results: dict[int, tuple[NDArray[np.float64], NDArray[np.float64], float, float]] = {}
        for arena_id in self.arena_ids:
            result_view = self._ipc.get_result_view(arena_id)
            results[arena_id] = (
                result_view.bid_depth.copy(),
                result_view.ask_depth.copy(),
                result_view.last_price,
                result_view.mid_price,
            )

        # 重置状态
        self._sync.reset_all_status(self.arena_ids)

        return results

    def shutdown(self) -> None:
        """关闭所有 Worker 并清理共享内存"""
        from src.training.arena.shared_memory_ipc import CommandStatus, CommandType

        if not self._started:
            return

        self.logger.info("正在关闭共享内存版 Worker 池...")

        # 发送 shutdown 命令到所有竞技场（只需要发送给每个 Worker 的第一个竞技场）
        shutdown_arena_ids: list[int] = []
        for worker_id in range(self.num_workers):
            # 找到该 Worker 负责的第一个竞技场
            for arena_id, worker_idx in self._arena_to_worker.items():
                if worker_idx == worker_id:
                    cmd_view = self._ipc.get_command_view(arena_id)
                    cmd_view.cmd_type = CommandType.SHUTDOWN
                    cmd_view.status = CommandStatus.PENDING
                    shutdown_arena_ids.append(arena_id)
                    break

        # 等待 Worker 退出（不需要等待 DONE，因为 Worker 会退出）
        for worker in self._workers:
            worker.join(timeout=5.0)
            if worker.is_alive():
                self.logger.warning(f"Worker {worker.pid} 未能正常退出，强制终止")
                worker.terminate()

        # 清理共享内存
        self._ipc.close()
        self._ipc.unlink()

        self._workers.clear()
        self._ready_events.clear()
        self._started = False
        self.logger.info("共享内存版 Worker 池已关闭")

    def __enter__(self) -> "ArenaExecuteWorkerPoolShm":
        """上下文管理器入口"""
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """上下文管理器退出"""
        self.shutdown()


def _read_result_from_shm(
    arena_id: int,
    result_view: Any,  # ArenaResultView
) -> ArenaExecuteResult:
    """从共享内存读取执行结果

    Args:
        arena_id: 竞技场 ID
        result_view: 共享内存结果视图

    Returns:
        执行结果
    """
    return ArenaExecuteResult(
        arena_id=arena_id,
        bid_depth=result_view.bid_depth.copy(),
        ask_depth=result_view.ask_depth.copy(),
        last_price=result_view.last_price,
        mid_price=result_view.mid_price,
        trades=result_view.get_trades(),
        pending_updates=result_view.get_pending_updates(),
        mm_order_updates=result_view.get_mm_order_updates(),
    )
