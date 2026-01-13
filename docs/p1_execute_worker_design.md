# P1 Execute Worker Pool 设计文档

## 目标

将 Execute 阶段（约 2800ms/tick，占总时间 56%）并行化，通过持久 Worker 池实现竞技场级别的并行执行。

## 架构设计

```
┌─────────────────────────────────────────────────────────────────┐
│                        主进程                                    │
│  BatchNetworkCache + AgentAccountState + 协调                    │
│                                                                 │
│  每个 tick:                                                      │
│  1. 从 Worker 接收：订单簿深度 + 上一 tick 成交结果              │
│  2. 更新 AgentAccountState                                      │
│  3. 强平检测 + 推理 + 生成决策                                  │
│  4. 发送给 Worker：决策 + 强平 Agent 列表                       │
└─────────────────────────────────────────────────────────────────┘
                              ↕ Queue
┌─────────────────────────────────────────────────────────────────┐
│                        Worker 池                                 │
│  每个 Worker 维护若干个竞技场的 OrderBook                        │
│                                                                 │
│  每个 tick:                                                      │
│  1. 接收：决策 + 强平 Agent 列表                                │
│  2. 执行强平（撤单 + 市价单）                                   │
│  3. 执行 Agent 决策                                             │
│  4. 返回：订单簿深度 + 成交列表                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 数据结构

### ExecuteCommand (主进程 → Worker)

```python
@dataclass
class ExecuteCommand:
    cmd_type: str  # "reset", "init_mm", "execute", "get_depth", "shutdown"
    arena_id: int
    data: Any  # 命令特定数据
```

### ArenaExecuteData (execute 命令的 data)

```python
@dataclass
class ArenaExecuteData:
    # 强平相关
    liquidated_agents: list[tuple[int, int, int]]  # (agent_id, position_qty, is_mm)
    # 决策列表
    decisions: list[tuple[int, int, int, float, int]]  # (agent_id, action_int, side_int, price, quantity)
    # 做市商决策
    mm_decisions: list[tuple[int, list, list]]  # (agent_id, bid_orders, ask_orders)
```

### ArenaExecuteResult (Worker → 主进程)

```python
@dataclass
class ArenaExecuteResult:
    arena_id: int
    # 订单簿深度
    bid_depth: np.ndarray  # shape (100, 2) - (price, quantity)
    ask_depth: np.ndarray  # shape (100, 2)
    last_price: float
    mid_price: float
    # 成交列表
    trades: list[tuple]  # (trade_id, price, qty, buyer_id, seller_id, buyer_fee, seller_fee, is_buyer_taker)
    # 挂单状态更新
    pending_updates: dict[int, int | None]  # agent_id -> pending_order_id
    mm_order_updates: dict[int, tuple[list, list]]  # agent_id -> (bid_ids, ask_ids)
```

## 核心类

### ArenaExecuteWorkerPool

```python
class ArenaExecuteWorkerPool:
    """竞技场执行 Worker 池"""

    def __init__(
        self,
        num_workers: int,
        arena_ids: list[int],
        config: Config,
    ):
        """
        Args:
            num_workers: Worker 进程数量（建议 4-8）
            arena_ids: 所有竞技场 ID 列表
            config: 全局配置
        """

    def reset_all(self) -> None:
        """重置所有竞技场的订单簿"""

    def init_market_makers(
        self,
        mm_init_orders: dict[int, list[tuple[int, list, list]]]
    ) -> dict[int, ArenaExecuteResult]:
        """初始化做市商挂单（Episode 开始时调用）"""

    def execute_all(
        self,
        arena_commands: dict[int, ArenaExecuteData]
    ) -> dict[int, ArenaExecuteResult]:
        """执行所有竞技场的决策（每个 tick 调用）"""

    def get_all_depths(self) -> dict[int, tuple[np.ndarray, np.ndarray, float, float]]:
        """获取所有竞技场的订单簿深度"""

    def shutdown(self) -> None:
        """关闭所有 Worker"""
```

### Worker 进程函数

```python
def arena_execute_worker(
    worker_id: int,
    arena_ids: list[int],
    config: Config,
    cmd_queue: Queue,
    result_queue: Queue,
) -> None:
    """Worker 进程主函数"""
```

## 状态管理

### 主进程维护

- `AgentAccountState`: 所有 Agent 的账户状态（余额、持仓、已实现盈亏）
- `network_caches`: 神经网络缓存
- 强平逻辑、ADL 逻辑

### Worker 维护

- `MatchingEngine` / `OrderBook`: 订单簿和撮合引擎
- `pending_order_id`: 非做市商的挂单 ID（用于撤单）
- `bid_order_ids`, `ask_order_ids`: 做市商的挂单 ID 列表
- `recent_trades`: 最近成交记录

### 状态同步

成交后需要同步的数据：
1. **主进程更新 AgentAccountState**：根据 trades 更新 balance, position, realized_pnl
2. **Worker 更新挂单状态**：返回 pending_updates, mm_order_updates 给主进程
3. **主进程更新挂单 ID 到 AgentAccountState**

## 通信优化

### 数据量估算

- 决策数据：~5 MB / tick（25 竞技场 × 10600 Agent）
- 深度数据：~40 KB / tick（25 竞技场 × 400 float）
- 成交数据：~100 KB / tick（假设 5000 笔成交）
- 总计：~5.2 MB / tick

### 预期延迟

- 序列化 + 反序列化：~100 ms
- 并行执行：~150 ms（原 2800ms / 16-20 Worker）
- 总计：~250 ms（比原来 2800ms 快约 10x）

## 实现步骤

1. 创建 `src/training/arena/execute_worker.py`
2. 实现 `ArenaExecuteWorkerPool` 类
3. 实现 Worker 进程函数
4. 修改 `ParallelArenaTrainer.run_tick_all_arenas()` 使用 Worker 池
5. 处理 Episode 重置和做市商初始化
6. 处理强平和 ADL 逻辑

## 注意事项

1. **ADL 处理**：ADL 不需要订单簿操作，完全在主进程执行
2. **鲶鱼行动**：需要在 Worker 中执行，因为涉及订单簿操作
3. **Episode 重置**：需要同时重置主进程的 AgentAccountState 和 Worker 的 OrderBook
4. **错误处理**：Worker 异常时需要能够恢复
