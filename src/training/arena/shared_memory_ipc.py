"""共享内存 IPC 模块

本模块实现基于 multiprocessing.shared_memory 的零拷贝 IPC 机制，
替代现有的 Queue + pickle 方式，减少进程间数据传输开销。

主要组件:
- CommandStatus: 命令状态枚举
- CommandType: 命令类型枚举
- ArenaCommandView: 单个竞技场命令区域的零拷贝视图
- ArenaResultView: 单个竞技场结果区域的零拷贝视图
- SharedMemoryIPC: 共享内存 IPC 管理器
- ShmSynchronizer: 共享内存同步器（无锁轮询）
"""

from __future__ import annotations

import atexit
import logging
import time
import uuid
from enum import IntEnum
from multiprocessing import shared_memory
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from src.training.arena.execute_worker import (
        CatfishDecision,
    )

# ============================================================================
# 常量定义
# ============================================================================

MAX_ARENAS = 64              # 最大竞技场数
MAX_LIQUIDATED = 500         # 每竞技场最大强平数
MAX_DECISIONS = 12000        # 每竞技场最大决策数
MAX_MM_AGENTS = 500          # 每竞技场最大做市商数
MAX_ORDERS_PER_MM = 20       # 每做市商最大订单数
MAX_TRADES = 2000            # 每竞技场最大成交数
MAX_PENDING_UPDATES = 12000  # 最大挂单更新数

# 鲶鱼相关常量
MAX_CATFISH = 3              # 最大鲶鱼数（3种鲶鱼：趋势创造者、均值回归、随机交易）

# Header 大小（64 bytes，填充到缓存行对齐）
HEADER_SIZE = 64

# 计算各数据区域大小
LIQUIDATED_SIZE = MAX_LIQUIDATED * 3 * 8  # int64[MAX_LIQUIDATED × 3]
DECISIONS_SIZE = MAX_DECISIONS * 5 * 8    # float64[MAX_DECISIONS × 5]
MM_AGENT_IDS_SIZE = MAX_MM_AGENTS * 8     # int64[MAX_MM_AGENTS] - 存储做市商 agent_id
MM_DECISIONS_SIZE = MAX_MM_AGENTS * MAX_ORDERS_PER_MM * 2 * 2 * 8  # float64[..., 4]

# 吃单鲶鱼决策: catfish_id(int64), direction(int64), quantity_ticks(int64)
CATFISH_DECISIONS_SIZE = MAX_CATFISH * 3 * 8  # int64[MAX_CATFISH × 3]

# Command Region 总大小（包含鲶鱼决策区域）
COMMAND_REGION_SIZE = (
    HEADER_SIZE
    + LIQUIDATED_SIZE
    + DECISIONS_SIZE
    + MM_AGENT_IDS_SIZE
    + MM_DECISIONS_SIZE
    + CATFISH_DECISIONS_SIZE
)

# Result Region 各区域大小
DEPTH_SIZE = 100 * 2 * 8  # float64[100, 2]
TRADES_SIZE = MAX_TRADES * 8 * 8  # float64[MAX_TRADES × 8]
PENDING_UPDATES_SIZE = MAX_PENDING_UPDATES * 2 * 8  # int64[MAX_PENDING_UPDATES × 2]
MM_ORDER_IDS_SIZE = MAX_MM_AGENTS * (1 + MAX_ORDERS_PER_MM * 2) * 8  # int64[...]

# Result Region 总大小
RESULT_REGION_SIZE = HEADER_SIZE + DEPTH_SIZE * 2 + TRADES_SIZE + PENDING_UPDATES_SIZE + MM_ORDER_IDS_SIZE

# 总共享内存大小
TOTAL_SHM_SIZE = MAX_ARENAS * (COMMAND_REGION_SIZE + RESULT_REGION_SIZE)

logger = logging.getLogger(__name__)


# ============================================================================
# 枚举定义
# ============================================================================


class CommandStatus(IntEnum):
    """命令状态枚举"""
    IDLE = 0        # 空闲
    PENDING = 1     # 待处理
    PROCESSING = 2  # 处理中
    DONE = 3        # 完成


class CommandType(IntEnum):
    """命令类型枚举"""
    RESET = 0       # 重置
    INIT_MM = 1     # 初始化做市商
    EXECUTE = 2     # 执行订单
    GET_DEPTH = 3   # 获取深度
    SHUTDOWN = 4    # 关闭


# ============================================================================
# Command View
# ============================================================================


class ArenaCommandView:
    """单个竞技场命令区域的零拷贝视图

    内存布局（COMMAND_REGION_SIZE bytes）:
    - Header (64 bytes):
        - status: uint32 (4 bytes)
        - cmd_type: uint32 (4 bytes)
        - liquidated_count: uint32 (4 bytes)
        - decisions_count: uint32 (4 bytes)
        - mm_count: uint32 (4 bytes)
        - catfish_count: uint32 (4 bytes)
        - padding: 40 bytes
    - liquidated_data: int64[MAX_LIQUIDATED × 3]
    - decisions_data: float64[MAX_DECISIONS × 5]
    - mm_agent_ids: int64[MAX_MM_AGENTS] - 做市商 agent_id 列表
    - mm_decisions_data: float64[MAX_MM_AGENTS × MAX_ORDERS_PER_MM × 2 × 2]
    - catfish_decisions: int64[MAX_CATFISH × 3] - 吃单鲶鱼决策

    Attributes:
        _buffer: 底层内存视图
        _offset: 在共享内存中的偏移量
        _header: Header 区域的 numpy 视图
        _liquidated: 强平数据的 numpy 视图
        _decisions: 决策数据的 numpy 视图
        _mm_agent_ids: 做市商 agent_id 的 numpy 视图
        _mm_decisions: 做市商决策的 numpy 视图
        _catfish_decisions: 吃单鲶鱼决策的 numpy 视图

    """

    _header: NDArray[np.uint32]
    _liquidated: NDArray[np.int64]
    _decisions: NDArray[np.float64]
    _mm_agent_ids: NDArray[np.int64]
    _mm_decisions: NDArray[np.float64]
    _catfish_decisions: NDArray[np.int64]

    def __init__(self, buffer: bytes, offset: int = 0) -> None:
        """初始化命令视图

        Args:
            buffer: 共享内存缓冲区
            offset: 在缓冲区中的偏移量
        """
        self._buffer = buffer
        self._offset = offset

        # 计算各区域的偏移量
        header_offset = offset
        liquidated_offset = header_offset + HEADER_SIZE
        decisions_offset = liquidated_offset + LIQUIDATED_SIZE
        mm_agent_ids_offset = decisions_offset + DECISIONS_SIZE
        mm_decisions_offset = mm_agent_ids_offset + MM_AGENT_IDS_SIZE
        catfish_decisions_offset = mm_decisions_offset + MM_DECISIONS_SIZE

        # 创建 numpy 视图
        self._header = np.ndarray(
            shape=(16,), dtype=np.uint32, buffer=buffer, offset=header_offset
        )
        self._liquidated = np.ndarray(
            shape=(MAX_LIQUIDATED, 3), dtype=np.int64, buffer=buffer, offset=liquidated_offset
        )
        self._decisions = np.ndarray(
            shape=(MAX_DECISIONS, 5), dtype=np.float64, buffer=buffer, offset=decisions_offset
        )
        self._mm_agent_ids = np.ndarray(
            shape=(MAX_MM_AGENTS,), dtype=np.int64, buffer=buffer, offset=mm_agent_ids_offset
        )
        self._mm_decisions = np.ndarray(
            shape=(MAX_MM_AGENTS, MAX_ORDERS_PER_MM * 2, 2), dtype=np.float64,
            buffer=buffer, offset=mm_decisions_offset
        )
        self._catfish_decisions = np.ndarray(
            shape=(MAX_CATFISH, 3), dtype=np.int64, buffer=buffer, offset=catfish_decisions_offset
        )

    def release(self) -> None:
        """释放所有 numpy 视图引用

        在关闭共享内存前调用，避免 BufferError。
        """
        # 将 numpy 数组设为 None，释放对 buffer 的引用
        self._header = None  # type: ignore[assignment]
        self._liquidated = None  # type: ignore[assignment]
        self._decisions = None  # type: ignore[assignment]
        self._mm_agent_ids = None  # type: ignore[assignment]
        self._mm_decisions = None  # type: ignore[assignment]
        self._catfish_decisions = None  # type: ignore[assignment]

    @property
    def status(self) -> int:
        """获取命令状态"""
        return int(self._header[0])

    @status.setter
    def status(self, value: int) -> None:
        """设置命令状态"""
        self._header[0] = value

    @property
    def cmd_type(self) -> int:
        """获取命令类型"""
        return int(self._header[1])

    @cmd_type.setter
    def cmd_type(self, value: int) -> None:
        """设置命令类型"""
        self._header[1] = value

    @property
    def liquidated_count(self) -> int:
        """获取强平数量"""
        return int(self._header[2])

    @property
    def decisions_count(self) -> int:
        """获取决策数量"""
        return int(self._header[3])

    @property
    def mm_count(self) -> int:
        """获取做市商数量"""
        return int(self._header[4])

    @property
    def catfish_count(self) -> int:
        """获取吃单鲶鱼数量"""
        return int(self._header[5])

    def set_liquidated(
        self, agents: list[tuple[int, int, bool]]
    ) -> None:
        """设置强平数据（向量化优化版）

        Args:
            agents: 强平 Agent 列表，格式: [(agent_id, position_qty, is_mm), ...]
        """
        count = min(len(agents), MAX_LIQUIDATED)
        self._header[2] = count
        if count > 0:
            # 将 bool 转换为 int，然后一次性写入
            data = [(a[0], a[1], 1 if a[2] else 0) for a in agents[:count]]
            arr = np.array(data, dtype=np.int64)
            self._liquidated[:count] = arr

    def set_decisions(
        self, decisions: list[tuple[int, int, int, float, int]]
    ) -> None:
        """设置决策数据（向量化优化版）

        Args:
            decisions: 决策列表，格式: [(agent_id, action_int, side_int, price, quantity), ...]
        """
        count = min(len(decisions), MAX_DECISIONS)
        self._header[3] = count
        if count > 0:
            # 使用 NumPy 向量化操作，一次性写入所有决策
            arr = np.array(decisions, dtype=np.float64)
            self._decisions[:count] = arr

    def set_decisions_array(self, arr: np.ndarray) -> None:
        """直接设置决策数组（零拷贝优化版）

        用于共享内存优化，直接接受 NumPy 数组，避免 Python 列表转换开销。

        Args:
            arr: 决策数组，shape (N, 5)，列顺序: [agent_id, action_int, side_int, price, quantity]
        """
        count = min(arr.shape[0], MAX_DECISIONS)
        self._header[3] = count
        if count > 0:
            self._decisions[:count] = arr[:count]

    def set_mm_decisions(
        self, mm_decisions: list[tuple[int, list[dict[str, float]], list[dict[str, float]]]]
    ) -> None:
        """设置做市商决策数据

        Args:
            mm_decisions: 做市商决策列表，
                格式: [(agent_id, bid_orders, ask_orders), ...]
                其中 bid_orders/ask_orders: [{"price": float, "quantity": float}, ...]
        """
        count = min(len(mm_decisions), MAX_MM_AGENTS)
        self._header[4] = count

        # 重置 mm_decisions 区域（使用 -1 标记无效条目）
        self._mm_decisions[:count, :, :] = -1.0

        for i in range(count):
            agent_id, bid_orders, ask_orders = mm_decisions[i]

            # 存储 agent_id 到专用数组
            self._mm_agent_ids[i] = agent_id

            # 存储买单
            for j, order in enumerate(bid_orders[:MAX_ORDERS_PER_MM]):
                self._mm_decisions[i, j, 0] = order.get("price", 0.0)
                self._mm_decisions[i, j, 1] = order.get("quantity", 0.0)

            # 存储卖单（偏移 MAX_ORDERS_PER_MM）
            for j, order in enumerate(ask_orders[:MAX_ORDERS_PER_MM]):
                self._mm_decisions[i, MAX_ORDERS_PER_MM + j, 0] = order.get("price", 0.0)
                self._mm_decisions[i, MAX_ORDERS_PER_MM + j, 1] = order.get("quantity", 0.0)

    def get_liquidated(self) -> list[tuple[int, int, bool]]:
        """获取强平数据（向量化优化版）

        Returns:
            强平 Agent 列表，格式: [(agent_id, position_qty, is_mm), ...]
        """
        count = self.liquidated_count
        if count == 0:
            return []
        # 使用 NumPy 切片一次性读取
        data = self._liquidated[:count]
        return [(int(row[0]), int(row[1]), bool(row[2])) for row in data]

    def get_decisions(self) -> list[tuple[int, int, int, float, int]]:
        """获取决策数据（向量化优化版）

        Returns:
            决策列表，格式: [(agent_id, action_int, side_int, price, quantity), ...]
        """
        count = self.decisions_count
        if count == 0:
            return []
        # 使用 NumPy 切片一次性读取，然后转换为元组列表
        data = self._decisions[:count]
        return [
            (int(row[0]), int(row[1]), int(row[2]), row[3], int(row[4]))
            for row in data
        ]

    def get_mm_decisions(
        self,
    ) -> list[tuple[int, list[dict[str, float]], list[dict[str, float]]]]:
        """获取做市商决策数据

        Returns:
            做市商决策列表，格式: [(agent_id, bid_orders, ask_orders), ...]
        """
        count = self.mm_count
        result: list[tuple[int, list[dict[str, float]], list[dict[str, float]]]] = []

        for i in range(count):
            # 从专用数组读取 agent_id
            agent_id = int(self._mm_agent_ids[i])

            bid_orders: list[dict[str, float]] = []
            ask_orders: list[dict[str, float]] = []

            # 读取买单
            for j in range(MAX_ORDERS_PER_MM):
                price = self._mm_decisions[i, j, 0]
                quantity = self._mm_decisions[i, j, 1]
                if price >= 0 and quantity > 0:
                    bid_orders.append({"price": price, "quantity": quantity})

            # 读取卖单
            for j in range(MAX_ORDERS_PER_MM):
                price = self._mm_decisions[i, MAX_ORDERS_PER_MM + j, 0]
                quantity = self._mm_decisions[i, MAX_ORDERS_PER_MM + j, 1]
                if price >= 0 and quantity > 0:
                    ask_orders.append({"price": price, "quantity": quantity})

            result.append((agent_id, bid_orders, ask_orders))

        return result

    def set_catfish_decisions(
        self, decisions: list["CatfishDecision"]
    ) -> None:
        """设置吃单鲶鱼决策数据

        Args:
            decisions: 吃单鲶鱼决策列表，CatfishDecision 对象
        """
        count = min(len(decisions), MAX_CATFISH)
        self._header[5] = count
        if count > 0:
            for i, decision in enumerate(decisions[:count]):
                self._catfish_decisions[i, 0] = decision.catfish_id
                self._catfish_decisions[i, 1] = decision.direction
                self._catfish_decisions[i, 2] = decision.quantity_ticks

    def get_catfish_decisions(self) -> list["CatfishDecision"]:
        """获取吃单鲶鱼决策数据

        Returns:
            吃单鲶鱼决策列表，CatfishDecision 对象
        """
        from src.training.arena.execute_worker import CatfishDecision

        count = self.catfish_count
        if count == 0:
            return []

        result: list[CatfishDecision] = []
        for i in range(count):
            catfish_id = int(self._catfish_decisions[i, 0])
            direction = int(self._catfish_decisions[i, 1])
            quantity_ticks = int(self._catfish_decisions[i, 2])
            result.append(
                CatfishDecision(
                    catfish_id=catfish_id,
                    direction=direction,
                    quantity_ticks=quantity_ticks,
                )
            )
        return result

class ArenaResultView:
    """单个竞技场结果区域的零拷贝视图

    内存布局（RESULT_REGION_SIZE bytes）:
    - Header (64 bytes):
        - status: uint32 (4 bytes)
        - last_price: float64 (8 bytes)
        - mid_price: float64 (8 bytes)
        - trades_count: uint32 (4 bytes)
        - pending_count: uint32 (4 bytes)
        - mm_count: uint32 (4 bytes)
        - padding: 32 bytes
    - bid_depth: float64[100, 2]
    - ask_depth: float64[100, 2]
    - trades_data: float64[MAX_TRADES, 8]
    - pending_updates: int64[MAX_PENDING_UPDATES, 2]
    - mm_order_ids: int64[MAX_MM_AGENTS, 1 + MAX_ORDERS_PER_MM * 2]

    Attributes:
        _buffer: 底层内存视图
        _offset: 在共享内存中的偏移量
    """

    _header_bytes: NDArray[np.uint8]
    _status_view: NDArray[np.uint32]
    _prices_view: NDArray[np.float64]
    _counts_view: NDArray[np.uint32]
    _bid_depth: NDArray[np.float64]
    _ask_depth: NDArray[np.float64]
    _trades: NDArray[np.float64]
    _pending: NDArray[np.int64]
    _mm_order_ids: NDArray[np.int64]

    def __init__(self, buffer: memoryview, offset: int) -> None:
        """初始化结果视图

        Args:
            buffer: 共享内存的 memoryview
            offset: 本竞技场结果区域在共享内存中的起始偏移量
        """
        self._buffer = buffer
        self._offset = offset

        # 计算各区域偏移
        header_offset = offset
        bid_depth_offset = header_offset + HEADER_SIZE
        ask_depth_offset = bid_depth_offset + DEPTH_SIZE
        trades_offset = ask_depth_offset + DEPTH_SIZE
        pending_offset = trades_offset + TRADES_SIZE
        mm_order_ids_offset = pending_offset + PENDING_UPDATES_SIZE

        # 创建 numpy 视图（零拷贝）
        # Header 使用 bytes 视图，手动解析
        self._header_bytes = np.ndarray(
            shape=(HEADER_SIZE,),
            dtype=np.uint8,
            buffer=buffer,
            offset=header_offset,
        )

        # 创建专用的 header 视图
        self._status_view = np.ndarray(
            shape=(1,), dtype=np.uint32, buffer=buffer, offset=header_offset
        )
        self._prices_view = np.ndarray(
            shape=(2,), dtype=np.float64, buffer=buffer, offset=header_offset + 8
        )
        self._counts_view = np.ndarray(
            shape=(3,), dtype=np.uint32, buffer=buffer, offset=header_offset + 24
        )

        self._bid_depth = np.ndarray(
            shape=(100, 2),
            dtype=np.float64,
            buffer=buffer,
            offset=bid_depth_offset,
        )
        self._ask_depth = np.ndarray(
            shape=(100, 2),
            dtype=np.float64,
            buffer=buffer,
            offset=ask_depth_offset,
        )
        self._trades = np.ndarray(
            shape=(MAX_TRADES, 8),
            dtype=np.float64,
            buffer=buffer,
            offset=trades_offset,
        )
        self._pending = np.ndarray(
            shape=(MAX_PENDING_UPDATES, 2),
            dtype=np.int64,
            buffer=buffer,
            offset=pending_offset,
        )
        self._mm_order_ids = np.ndarray(
            shape=(MAX_MM_AGENTS, 1 + MAX_ORDERS_PER_MM * 2),
            dtype=np.int64,
            buffer=buffer,
            offset=mm_order_ids_offset,
        )

    def release(self) -> None:
        """释放所有 numpy 视图引用

        在关闭共享内存前调用，避免 BufferError。
        """
        # 将 numpy 数组设为 None，释放对 buffer 的引用
        self._header_bytes = None  # type: ignore[assignment]
        self._status_view = None  # type: ignore[assignment]
        self._prices_view = None  # type: ignore[assignment]
        self._counts_view = None  # type: ignore[assignment]
        self._bid_depth = None  # type: ignore[assignment]
        self._ask_depth = None  # type: ignore[assignment]
        self._trades = None  # type: ignore[assignment]
        self._pending = None  # type: ignore[assignment]
        self._mm_order_ids = None  # type: ignore[assignment]

    @property
    def status(self) -> int:
        """获取结果状态"""
        return int(self._status_view[0])

    @status.setter
    def status(self, value: int) -> None:
        """设置结果状态"""
        self._status_view[0] = value

    @property
    def bid_depth(self) -> NDArray[np.float64]:
        """获取买盘深度（零拷贝视图）"""
        return self._bid_depth

    @property
    def ask_depth(self) -> NDArray[np.float64]:
        """获取卖盘深度（零拷贝视图）"""
        return self._ask_depth

    @property
    def last_price(self) -> float:
        """获取最新价格"""
        return float(self._prices_view[0])

    @property
    def mid_price(self) -> float:
        """获取中间价"""
        return float(self._prices_view[1])

    @property
    def trades_count(self) -> int:
        """获取成交数量"""
        return int(self._counts_view[0])

    @property
    def pending_count(self) -> int:
        """获取挂单更新数量"""
        return int(self._counts_view[1])

    @property
    def mm_count(self) -> int:
        """获取做市商数量"""
        return int(self._counts_view[2])

    def set_depth(
        self, bid: NDArray[np.float64], ask: NDArray[np.float64]
    ) -> None:
        """设置订单簿深度

        Args:
            bid: 买盘深度，shape (100, 2)
            ask: 卖盘深度，shape (100, 2)
        """
        np.copyto(self._bid_depth, bid)
        np.copyto(self._ask_depth, ask)

    def set_prices(self, last_price: float, mid_price: float) -> None:
        """设置价格

        Args:
            last_price: 最新价格
            mid_price: 中间价
        """
        self._prices_view[0] = last_price
        self._prices_view[1] = mid_price

    def set_trades(
        self,
        trades: list[tuple[int, float, int, int, int, float, float, bool]],
    ) -> None:
        """设置成交数据（向量化优化版）

        Args:
            trades: 成交列表，每个元素格式:
                (trade_id, price, quantity, buyer_id, seller_id,
                 buyer_fee, seller_fee, is_buyer_taker)
        """
        count = min(len(trades), MAX_TRADES)
        self._counts_view[0] = count
        if count > 0:
            # 预处理数据，将 bool 转换为 float
            data = [
                (
                    float(t[0]), t[1], float(t[2]), float(t[3]),
                    float(t[4]), t[5], t[6], 1.0 if t[7] else 0.0
                )
                for t in trades[:count]
            ]
            arr = np.array(data, dtype=np.float64)
            self._trades[:count] = arr

    def set_pending_updates(self, updates: dict[int, int | None]) -> None:
        """设置挂单更新

        Args:
            updates: 挂单更新映射，agent_id -> pending_order_id | None
        """
        count = min(len(updates), MAX_PENDING_UPDATES)
        self._counts_view[1] = count
        for i, (agent_id, order_id) in enumerate(updates.items()):
            if i >= count:
                break
            self._pending[i, 0] = agent_id
            self._pending[i, 1] = order_id if order_id is not None else -1

    def set_mm_order_updates(
        self,
        updates: dict[int, tuple[list[int], list[int]]],
    ) -> None:
        """设置做市商挂单更新

        Args:
            updates: 做市商挂单更新映射，
                agent_id -> (bid_order_ids, ask_order_ids)
        """
        count = min(len(updates), MAX_MM_AGENTS)
        self._counts_view[2] = count
        for i, (agent_id, (bid_ids, ask_ids)) in enumerate(updates.items()):
            if i >= count:
                break
            self._mm_order_ids[i, 0] = agent_id
            # 存储买单 ID
            for j, order_id in enumerate(bid_ids[:MAX_ORDERS_PER_MM]):
                self._mm_order_ids[i, 1 + j] = order_id
            # 填充剩余位置为 -1
            for j in range(len(bid_ids), MAX_ORDERS_PER_MM):
                self._mm_order_ids[i, 1 + j] = -1
            # 存储卖单 ID
            for j, order_id in enumerate(ask_ids[:MAX_ORDERS_PER_MM]):
                self._mm_order_ids[i, 1 + MAX_ORDERS_PER_MM + j] = order_id
            # 填充剩余位置为 -1
            for j in range(len(ask_ids), MAX_ORDERS_PER_MM):
                self._mm_order_ids[i, 1 + MAX_ORDERS_PER_MM + j] = -1

    def get_trades(
        self,
    ) -> list[tuple[int, float, int, int, int, float, float, bool]]:
        """获取成交数据（向量化优化版）

        Returns:
            成交列表，每个元素格式:
                (trade_id, price, quantity, buyer_id, seller_id,
                 buyer_fee, seller_fee, is_buyer_taker)
        """
        count = self.trades_count
        if count == 0:
            return []
        # 使用 NumPy 切片一次性读取
        data = self._trades[:count]
        return [
            (
                int(row[0]), row[1], int(row[2]), int(row[3]),
                int(row[4]), row[5], row[6], row[7] > 0.5
            )
            for row in data
        ]

    def get_pending_updates(self) -> dict[int, int | None]:
        """获取挂单更新

        Returns:
            挂单更新映射，agent_id -> pending_order_id | None
        """
        count = self.pending_count
        result: dict[int, int | None] = {}
        for i in range(count):
            agent_id = int(self._pending[i, 0])
            order_id = int(self._pending[i, 1])
            result[agent_id] = order_id if order_id >= 0 else None
        return result

    def get_mm_order_updates(self) -> dict[int, tuple[list[int], list[int]]]:
        """获取做市商挂单更新

        Returns:
            做市商挂单更新映射，agent_id -> (bid_order_ids, ask_order_ids)
        """
        count = self.mm_count
        result: dict[int, tuple[list[int], list[int]]] = {}
        for i in range(count):
            agent_id = int(self._mm_order_ids[i, 0])
            bid_ids: list[int] = []
            ask_ids: list[int] = []
            # 读取买单 ID
            for j in range(MAX_ORDERS_PER_MM):
                order_id = int(self._mm_order_ids[i, 1 + j])
                if order_id >= 0:
                    bid_ids.append(order_id)
            # 读取卖单 ID
            for j in range(MAX_ORDERS_PER_MM):
                order_id = int(self._mm_order_ids[i, 1 + MAX_ORDERS_PER_MM + j])
                if order_id >= 0:
                    ask_ids.append(order_id)
            result[agent_id] = (bid_ids, ask_ids)
        return result


# ============================================================================
# SharedMemoryIPC
# ============================================================================


class SharedMemoryIPC:
    """共享内存 IPC 管理器

    管理多个竞技场的命令和结果区域，提供零拷贝的数据访问。

    内存布局:
    - Arena 0 Command Region (COMMAND_REGION_SIZE bytes)
    - Arena 0 Result Region (RESULT_REGION_SIZE bytes)
    - Arena 1 Command Region
    - Arena 1 Result Region
    - ...

    Attributes:
        _num_arenas: 竞技场数量
        _shm_name: 共享内存名称
        _shm: 共享内存对象
        _buffer: 共享内存的 memoryview
        _command_views: 各竞技场的命令视图
        _result_views: 各竞技场的结果视图
        _is_creator: 是否是创建者（负责 unlink）
    """

    def __init__(
        self,
        num_arenas: int,
        shm_name: str | None = None,
        create: bool = False,
    ) -> None:
        """初始化共享内存 IPC

        Args:
            num_arenas: 竞技场数量（不能超过 MAX_ARENAS）
            shm_name: 共享内存名称（可选，默认自动生成）
            create: 是否创建共享内存（True 为创建者，False 为连接者）
        """
        if num_arenas > MAX_ARENAS:
            raise ValueError(f"num_arenas ({num_arenas}) 超过最大值 ({MAX_ARENAS})")

        self._num_arenas = num_arenas
        self._is_creator = create

        # 计算实际需要的共享内存大小
        arena_size = COMMAND_REGION_SIZE + RESULT_REGION_SIZE
        total_size = num_arenas * arena_size

        if create:
            # 创建共享内存
            self._shm_name = shm_name or f"arena_ipc_{uuid.uuid4().hex[:8]}"
            self._shm = shared_memory.SharedMemory(
                name=self._shm_name,
                create=True,
                size=total_size,
            )
            # 初始化内存为零
            buf = self._shm.buf
            assert buf is not None, "SharedMemory buffer should not be None after creation"
            buf[:] = b'\x00' * total_size
            logger.info(f"创建共享内存: {self._shm_name}, 大小: {total_size} bytes")

            # 注册清理函数
            atexit.register(self._cleanup)
        else:
            # 连接到已有共享内存
            if shm_name is None:
                raise ValueError("连接模式需要指定 shm_name")
            self._shm_name = shm_name
            self._shm = shared_memory.SharedMemory(name=shm_name, create=False)
            logger.info(f"连接共享内存: {self._shm_name}")

        shm_buf = self._shm.buf
        if shm_buf is None:
            raise RuntimeError("SharedMemory buffer should not be None")
        self._buffer = memoryview(shm_buf)
        self._closed = False

        # 创建各竞技场的视图
        self._command_views: list[ArenaCommandView] = []
        self._result_views: list[ArenaResultView] = []

        for i in range(num_arenas):
            base_offset = i * arena_size
            cmd_offset = base_offset
            result_offset = base_offset + COMMAND_REGION_SIZE

            self._command_views.append(ArenaCommandView(self._buffer, cmd_offset))
            self._result_views.append(ArenaResultView(self._buffer, result_offset))

    @property
    def shm_name(self) -> str:
        """获取共享内存名称"""
        return self._shm_name

    def get_command_view(self, arena_id: int) -> ArenaCommandView:
        """获取竞技场的命令视图

        Args:
            arena_id: 竞技场 ID

        Returns:
            命令视图

        Raises:
            IndexError: arena_id 超出范围
        """
        if arena_id < 0 or arena_id >= self._num_arenas:
            raise IndexError(f"arena_id ({arena_id}) 超出范围 [0, {self._num_arenas})")
        return self._command_views[arena_id]

    def get_result_view(self, arena_id: int) -> ArenaResultView:
        """获取竞技场的结果视图

        Args:
            arena_id: 竞技场 ID

        Returns:
            结果视图

        Raises:
            IndexError: arena_id 超出范围
        """
        if arena_id < 0 or arena_id >= self._num_arenas:
            raise IndexError(f"arena_id ({arena_id}) 超出范围 [0, {self._num_arenas})")
        return self._result_views[arena_id]

    def close(self) -> None:
        """关闭共享内存连接

        注意：必须先释放所有 numpy 视图的引用，否则会出现 BufferError。
        """
        # 先释放各 View 的 numpy 数组引用
        for view in self._command_views:
            view.release()
        for view in self._result_views:
            view.release()

        # 清空视图列表
        self._command_views.clear()
        self._result_views.clear()

        # 释放 memoryview
        try:
            self._buffer.release()
        except Exception:
            pass  # 忽略已释放的情况

        try:
            self._shm.close()
            logger.debug(f"关闭共享内存连接: {self._shm_name}")
        except Exception as e:
            logger.warning(f"关闭共享内存时出错: {e}")

    def unlink(self) -> None:
        """删除共享内存（仅创建者应调用）"""
        if self._is_creator:
            try:
                self._shm.unlink()
                logger.info(f"删除共享内存: {self._shm_name}")
            except Exception as e:
                logger.warning(f"删除共享内存时出错: {e}")

    def _cleanup(self) -> None:
        """清理函数（atexit 调用）"""
        self.close()
        if self._is_creator:
            self.unlink()

    def __enter__(self) -> "SharedMemoryIPC":
        """上下文管理器入口"""
        return self

    def __exit__(self, exc_type: type | None, exc_val: Exception | None, exc_tb: object) -> None:
        """上下文管理器退出"""
        self.close()
        if self._is_creator:
            self.unlink()


# ============================================================================
# ShmSynchronizer
# ============================================================================


class ShmSynchronizer:
    """共享内存同步器（无锁轮询）

    提供等待和重置状态的辅助方法。

    Attributes:
        _ipc: 共享内存 IPC 管理器
    """

    def __init__(self, ipc: SharedMemoryIPC) -> None:
        """初始化同步器

        Args:
            ipc: 共享内存 IPC 管理器
        """
        self._ipc = ipc

    def wait_all_done(
        self,
        arena_ids: list[int],
        timeout_ms: float = 5000,
        poll_interval_us: float = 100,
    ) -> bool:
        """等待所有竞技场完成

        使用无锁轮询方式等待所有指定竞技场的结果状态变为 DONE。
        采用自适应等待策略：先 busy-wait，多次未完成后才 sleep。

        Args:
            arena_ids: 需要等待的竞技场 ID 列表
            timeout_ms: 超时时间（毫秒）
            poll_interval_us: 轮询间隔（微秒）- 仅在长时间等待时使用

        Returns:
            True 如果所有竞技场都完成，False 如果超时
        """
        start_time = time.perf_counter()
        timeout_s = timeout_ms / 1000.0
        poll_interval_s = poll_interval_us / 1_000_000.0

        # 使用列表追踪未完成的竞技场，避免 set 的哈希开销
        pending_list = list(arena_ids)
        idle_loops = 0
        BUSY_WAIT_LOOPS = 1000  # busy-wait 循环次数

        while pending_list:
            # 检查是否超时
            if time.perf_counter() - start_time > timeout_s:
                logger.warning(f"等待超时，未完成的竞技场: {pending_list}")
                return False

            # 检查各竞技场状态，直接修改列表
            i = 0
            progress = False
            while i < len(pending_list):
                arena_id = pending_list[i]
                result_view = self._ipc.get_result_view(arena_id)
                if result_view.status == CommandStatus.DONE:
                    # 用最后一个元素替换当前位置，然后 pop
                    pending_list[i] = pending_list[-1]
                    pending_list.pop()
                    progress = True
                else:
                    i += 1

            if pending_list:
                if progress:
                    idle_loops = 0  # 有进展，重置计数
                else:
                    idle_loops += 1
                    if idle_loops >= BUSY_WAIT_LOOPS:
                        # 长时间无进展，开始 sleep
                        time.sleep(poll_interval_s)

        return True

    def reset_all_status(self, arena_ids: list[int]) -> None:
        """重置所有竞技场的状态

        将命令状态设为 IDLE，结果状态设为 IDLE。

        Args:
            arena_ids: 需要重置的竞技场 ID 列表
        """
        for arena_id in arena_ids:
            cmd_view = self._ipc.get_command_view(arena_id)
            result_view = self._ipc.get_result_view(arena_id)
            cmd_view.status = CommandStatus.IDLE
            result_view.status = CommandStatus.IDLE


# ============================================================================
# 测试代码
# ============================================================================


def _test_basic() -> None:
    """基本功能测试"""
    print("=== 基本功能测试 ===")

    # 测试创建和销毁
    ipc = SharedMemoryIPC(num_arenas=4, create=True)
    print(f"共享内存名称: {ipc.shm_name}")

    # 测试命令视图
    cmd_view = ipc.get_command_view(0)
    cmd_view.status = CommandStatus.PENDING
    cmd_view.cmd_type = CommandType.EXECUTE
    assert cmd_view.status == CommandStatus.PENDING
    assert cmd_view.cmd_type == CommandType.EXECUTE
    print("命令视图读写: OK")

    # 测试强平数据
    liquidated = [(100, 50, True), (200, -30, False)]
    cmd_view.set_liquidated(liquidated)
    result = cmd_view.get_liquidated()
    assert len(result) == 2
    assert result[0] == (100, 50, True)
    assert result[1] == (200, -30, False)
    print("强平数据读写: OK")

    # 测试决策数据
    decisions = [(1001, 1, 1, 10000.5, 10), (1002, 4, 2, 0.0, 20)]
    cmd_view.set_decisions(decisions)
    result_decisions = cmd_view.get_decisions()
    assert len(result_decisions) == 2
    assert result_decisions[0] == (1001, 1, 1, 10000.5, 10)
    print("决策数据读写: OK")

    # 测试结果视图
    result_view = ipc.get_result_view(0)
    result_view.set_prices(10000.0, 10001.5)
    assert result_view.last_price == 10000.0
    assert result_view.mid_price == 10001.5
    print("结果视图价格读写: OK")

    # 测试深度数据
    bid_depth = np.array([[10000.0, 100], [9999.0, 200]], dtype=np.float64)
    ask_depth = np.array([[10001.0, 150], [10002.0, 250]], dtype=np.float64)
    bid_full = np.zeros((100, 2), dtype=np.float64)
    ask_full = np.zeros((100, 2), dtype=np.float64)
    bid_full[:2] = bid_depth
    ask_full[:2] = ask_depth
    result_view.set_depth(bid_full, ask_full)
    assert result_view.bid_depth[0, 0] == 10000.0
    assert result_view.ask_depth[0, 0] == 10001.0
    print("深度数据读写: OK")

    # 测试成交数据
    trades = [(1, 10000.0, 10, 100, 200, 0.5, 0.3, True)]
    result_view.set_trades(trades)
    result_trades = result_view.get_trades()
    assert len(result_trades) == 1
    assert result_trades[0][1] == 10000.0
    print("成交数据读写: OK")

    # 测试挂单更新
    pending = {100: 12345, 200: None}
    result_view.set_pending_updates(pending)
    result_pending = result_view.get_pending_updates()
    assert result_pending[100] == 12345
    assert result_pending[200] is None
    print("挂单更新读写: OK")

    # 测试做市商订单更新
    mm_updates = {300: ([1, 2, 3], [4, 5])}
    result_view.set_mm_order_updates(mm_updates)
    result_mm = result_view.get_mm_order_updates()
    assert result_mm[300] == ([1, 2, 3], [4, 5])
    print("做市商订单更新读写: OK")

    # 清理
    ipc.close()
    ipc.unlink()
    print("清理: OK")
    print()


def _test_synchronizer() -> None:
    """同步器测试"""
    print("=== 同步器测试 ===")

    ipc = SharedMemoryIPC(num_arenas=4, create=True)
    sync = ShmSynchronizer(ipc)

    # 设置所有竞技场状态为 DONE
    for i in range(4):
        result_view = ipc.get_result_view(i)
        result_view.status = CommandStatus.DONE

    # 测试等待
    success = sync.wait_all_done([0, 1, 2, 3], timeout_ms=100)
    assert success
    print("等待完成: OK")

    # 测试重置
    sync.reset_all_status([0, 1, 2, 3])
    for i in range(4):
        cmd_view = ipc.get_command_view(i)
        result_view = ipc.get_result_view(i)
        assert cmd_view.status == CommandStatus.IDLE
        assert result_view.status == CommandStatus.IDLE
    print("重置状态: OK")

    # 清理
    ipc.close()
    ipc.unlink()
    print("清理: OK")
    print()


def _test_multiprocess() -> None:
    """多进程访问测试"""
    import multiprocessing

    print("=== 多进程访问测试 ===")

    # 创建共享内存
    ipc = SharedMemoryIPC(num_arenas=4, create=True)
    shm_name = ipc.shm_name

    def worker_func(shm_name: str, arena_id: int) -> None:
        """Worker 进程函数"""
        # 连接共享内存
        worker_ipc = SharedMemoryIPC(num_arenas=4, shm_name=shm_name, create=False)

        # 等待命令
        cmd_view = worker_ipc.get_command_view(arena_id)
        while cmd_view.status != CommandStatus.PENDING:
            time.sleep(0.001)

        # 处理命令
        cmd_view.status = CommandStatus.PROCESSING

        # 写入结果
        result_view = worker_ipc.get_result_view(arena_id)
        result_view.set_prices(10000.0 + arena_id, 10001.0 + arena_id)
        result_view.status = CommandStatus.DONE

        # 关闭连接
        worker_ipc.close()

    # 启动 Worker 进程
    workers = []
    for i in range(4):
        p = multiprocessing.Process(target=worker_func, args=(shm_name, i))
        p.start()
        workers.append(p)

    # 发送命令
    for i in range(4):
        cmd_view = ipc.get_command_view(i)
        cmd_view.cmd_type = CommandType.GET_DEPTH
        cmd_view.status = CommandStatus.PENDING

    # 等待完成
    sync = ShmSynchronizer(ipc)
    success = sync.wait_all_done([0, 1, 2, 3], timeout_ms=5000)
    assert success
    print("多进程等待完成: OK")

    # 验证结果
    for i in range(4):
        result_view = ipc.get_result_view(i)
        assert result_view.last_price == 10000.0 + i
        assert result_view.mid_price == 10001.0 + i
    print("多进程结果验证: OK")

    # 等待 Worker 退出
    for p in workers:
        p.join(timeout=2.0)

    # 清理
    ipc.close()
    ipc.unlink()
    print("清理: OK")
    print()


if __name__ == "__main__":
    _test_basic()
    _test_synchronizer()
    _test_multiprocess()
    print("所有测试通过！")
