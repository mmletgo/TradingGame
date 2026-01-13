# Arena 模块

## 模块概述

竞技场模块提供多竞技场并行推理训练功能。核心特性是将多个竞技场的神经网络推理合并成一个批量操作，只有交易配对和账户更新串行执行。

## 文件结构

- `__init__.py` - 模块导出
- `arena_state.py` - 竞技场状态类（AgentAccountState、CatfishAccountState、ArenaState）
- `execute_worker.py` - Execute Worker 池（并行化 Execute 阶段）
- `fitness_aggregator.py` - 适应度汇总器
- `parallel_arena_trainer.py` - 多竞技场并行推理训练器

## 核心类

### AgentAccountState (arena_state.py)

Agent 账户状态类，将 Agent 账户状态与 Agent 对象解耦，每个竞技场维护独立副本。

```python
@dataclass
class AgentAccountState:
    """Agent 账户状态（轻量级，约 200 bytes）"""
    agent_id: int
    agent_type: AgentType
    balance: float
    position_quantity: int
    position_avg_price: float
    realized_pnl: float
    leverage: float
    maintenance_margin_rate: float
    initial_balance: float
    pending_order_id: int | None
    maker_volume: int
    volatility_contribution: float
    is_liquidated: bool
    order_counter: int
    maker_fee_rate: float
    taker_fee_rate: float
    bid_order_ids: list[int]   # 做市商买单挂单 ID 列表
    ask_order_ids: list[int]   # 做市商卖单挂单 ID 列表
```

**主要方法：**

| 方法 | 描述 |
|------|------|
| `from_agent(agent)` | 类方法，从 Agent 对象创建状态副本 |
| `reset(config)` | 重置到初始状态 |
| `get_equity(current_price)` | 计算净值（余额 + 浮动盈亏） |
| `get_margin_ratio(current_price)` | 计算保证金率 |
| `check_liquidation(current_price)` | 检查是否需要强平 |
| `on_trade(trade_price, trade_quantity, is_buyer, fee, is_maker)` | 处理成交，返回已实现盈亏 |
| `generate_order_id(arena_id)` | 生成跨竞技场唯一的订单 ID |

### CatfishAccountState (arena_state.py)

鲶鱼账户状态类，包含鲶鱼特有的策略状态。

```python
@dataclass
class CatfishAccountState:
    """鲶鱼账户状态"""
    catfish_id: int
    catfish_mode: CatfishMode      # 鲶鱼类型
    balance: float
    position_quantity: int
    position_avg_price: float
    realized_pnl: float
    leverage: float
    maintenance_margin_rate: float
    initial_balance: float
    is_liquidated: bool
    order_counter: int
    current_direction: int         # 趋势创造者方向
    ema: float                     # 均值回归 EMA
    ema_initialized: bool
    last_action_tick: int
    phase_offset: int              # 相位偏移
    action_cooldown: int           # 行动冷却时间
    ma_period: int                 # 均线周期
    deviation_threshold: float     # 偏离阈值
    action_probability: float      # 随机交易触发概率
```

**主要方法：**

| 方法 | 描述 |
|------|------|
| `from_catfish(catfish)` | 类方法，从 CatfishBase 对象创建状态副本 |
| `reset(initial_balance)` | 重置到初始状态，趋势创造者会重新随机选择方向 |
| `decide(tick, price_history)` | 决策是否行动和方向，根据鲶鱼类型执行不同逻辑 |
| `can_act(tick)` | 检查冷却时间和相位偏移 |
| `record_action(tick)` | 记录行动时间 |
| `update_ema(price, ma_period)` | 更新 EMA 值（均值回归用） |

**鲶鱼类型决策逻辑：**

| 类型 | 决策逻辑 |
|------|---------|
| TREND_CREATOR | 每个 Episode 开始时随机选择方向，整个 Episode 保持该方向 |
| MEAN_REVERSION | 当价格偏离 EMA 超过阈值时反向操作 |
| RANDOM | 随机概率触发，方向也随机 |

**独立随机性保证：**
- 每个竞技场在 `reset()` 时独立随机选择趋势创造者方向
- 确保不同竞技场的行情有差异

### ArenaState (arena_state.py)

单个竞技场的独立状态，封装竞技场运行所需的所有状态。

```python
@dataclass
class ArenaState:
    """单个竞技场的独立状态"""
    arena_id: int
    matching_engine: MatchingEngine
    adl_manager: ADLManager
    agent_states: dict[int, AgentAccountState]
    catfish_states: dict[int, CatfishAccountState]
    recent_trades: deque
    price_history: list[float]
    tick_history_prices: list[float]
    tick_history_volumes: list[float]
    tick_history_amounts: list[float]
    smooth_mid_price: float
    tick: int
    pop_liquidated_counts: dict[AgentType, int]
    eliminating_agents: set[int]
    episode_high_price: float
    episode_low_price: float
    catfish_liquidated: bool
```

### FitnessAggregator (fitness_aggregator.py)

适应度汇总器，用于汇总多个竞技场、多个 episode 的适应度数据。

**方法：**
- `aggregate_simple_average(arena_fitnesses, episode_counts)` - 简单加权平均

**公式：**
```
avg_fitness = sum(arena_fitness) / total_episodes
```

---

### ArenaExecuteWorkerPool (execute_worker.py)

竞技场执行 Worker 池，将 Execute 阶段并行化。每个 Worker 维护若干个竞技场的 OrderBook 和 MatchingEngine。

**设计目标：**
- 将 Execute 阶段（约 2800ms/tick，占总时间 56%）并行化
- 通过持久 Worker 池实现竞技场级别的并行执行
- 预期延迟：~250ms（vs 原来 2800ms，快约 10x）

**架构：**
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

**数据类定义：**

```python
@dataclass
class ExecuteCommand:
    """主进程 -> Worker 的命令"""
    cmd_type: str  # "reset", "init_mm", "execute", "get_depth", "shutdown"
    arena_id: int
    data: Any

@dataclass
class ArenaExecuteData:
    """execute 命令的数据"""
    liquidated_agents: list[tuple[int, int, bool]]  # (agent_id, position_qty, is_mm)
    decisions: list[tuple[int, int, int, float, int]]  # (agent_id, action_int, side_int, price, quantity)
    mm_decisions: list[tuple[int, list, list]]  # (agent_id, bid_orders, ask_orders)

@dataclass
class ArenaExecuteResult:
    """Worker -> 主进程的结果"""
    arena_id: int
    bid_depth: np.ndarray  # shape (100, 2) - (price, quantity)
    ask_depth: np.ndarray  # shape (100, 2)
    last_price: float
    mid_price: float
    trades: list[tuple]  # (trade_id, price, qty, buyer_id, seller_id, buyer_fee, seller_fee, is_buyer_taker)
    pending_updates: dict[int, int | None]  # agent_id -> pending_order_id
    mm_order_updates: dict[int, tuple[list, list]]  # agent_id -> (bid_ids, ask_ids)
    error: str | None
```

**主要方法：**

| 方法 | 描述 |
|------|------|
| `start()` | 启动所有 Worker 进程 |
| `reset_all(initial_price, fee_rates)` | 重置所有竞技场的订单簿 |
| `init_market_makers(mm_init_orders)` | 初始化做市商挂单（Episode 开始时调用） |
| `execute_all(arena_commands)` | 执行所有竞技场的决策（每个 tick 调用） |
| `get_all_depths()` | 获取所有竞技场的订单簿深度 |
| `shutdown()` | 关闭所有 Worker |

**Worker 维护的状态：**
- `MatchingEngine` / `OrderBook`: 订单簿和撮合引擎
- `pending_order_ids`: 非做市商的挂单 ID（agent_id -> order_id）
- `mm_bid_order_ids`, `mm_ask_order_ids`: 做市商的挂单 ID 列表
- `order_counters`: 各 Agent 的订单计数器

**订单 ID 生成：**
使用全局唯一的订单 ID 格式：`(arena_id << 40) | (agent_id << 16) | order_counter`

**使用示例：**
```python
from src.training.arena import ArenaExecuteWorkerPool, ArenaExecuteData

# 创建 Worker 池
pool = ArenaExecuteWorkerPool(
    num_workers=4,
    arena_ids=[0, 1, 2, 3, 4, 5, 6, 7],
    config=config,
)

# 使用上下文管理器
with pool:
    # 重置所有竞技场
    pool.reset_all(initial_price=100.0, fee_rates={...})

    # 初始化做市商
    results = pool.init_market_makers({
        0: [(mm_id, bid_orders, ask_orders), ...],
        ...
    })

    # 每个 tick 执行
    results = pool.execute_all({
        0: ArenaExecuteData(
            liquidated_agents=[...],
            decisions=[...],
            mm_decisions=[...],
        ),
        ...
    })
```

---

### ArenaExecuteWorkerPoolShm (execute_worker.py)

共享内存版竞技场执行 Worker 池，使用 `SharedMemoryIPC` 替代 Queue 通信，减少序列化开销。

**设计目标：**
- 通过共享内存避免 Queue 的 pickle 序列化开销
- 使用无锁轮询实现低延迟通信
- 保持与 `ArenaExecuteWorkerPool` 相同的接口

**与 Queue 版的对比：**

| 特性 | ArenaExecuteWorkerPool (Queue) | ArenaExecuteWorkerPoolShm (Shm) |
|------|-------------------------------|--------------------------------|
| 通信方式 | Queue + pickle | SharedMemory + 轮询 |
| 序列化开销 | 高（每次通信都需要序列化） | 零（直接内存访问） |
| 同步机制 | Queue.put/get 阻塞 | 无锁轮询 + 状态标志 |
| 内存拷贝 | 每次通信需要拷贝 | 读取结果时需要 copy() |
| 适用场景 | 数据量小、通信不频繁 | 高频通信、大数据量 |

**架构：**
```
┌─────────────────────────────────────────────────────────────────┐
│                        主进程                                    │
│  SharedMemoryIPC (create=True) + ShmSynchronizer                │
│                                                                 │
│  每个 tick:                                                      │
│  1. 写入命令到 ArenaCommandView (set_decisions 等)              │
│  2. 设置 status = PENDING                                       │
│  3. 调用 wait_all_done() 轮询等待                               │
│  4. 从 ArenaResultView 读取结果                                 │
│  5. 调用 reset_all_status() 重置状态                            │
└─────────────────────────────────────────────────────────────────┘
                         ↕ SharedMemory
┌─────────────────────────────────────────────────────────────────┐
│                        Worker 池                                 │
│  SharedMemoryIPC (create=False) + 轮询循环                       │
│                                                                 │
│  每个 Worker 循环:                                               │
│  1. 轮询检查 status == PENDING                                  │
│  2. 设置 status = PROCESSING                                    │
│  3. 从 ArenaCommandView 读取数据执行                            │
│  4. 写入结果到 ArenaResultView                                  │
│  5. 设置 status = DONE                                          │
└─────────────────────────────────────────────────────────────────┘
```

**主要组件：**
- `arena_execute_worker_shm()`: Worker 进程主函数，使用共享内存通信
- `ArenaExecuteWorkerPoolShm`: Worker 池管理类
- `_write_result_to_shm()`: 将执行结果写入共享内存
- `_read_result_from_shm()`: 从共享内存读取执行结果

**主要方法（与 ArenaExecuteWorkerPool 接口兼容）：**

| 方法 | 描述 |
|------|------|
| `start()` | 启动所有 Worker 进程，等待 ready_event |
| `reset_all(initial_price, fee_rates)` | 重置所有竞技场的订单簿 |
| `init_market_makers(mm_init_orders)` | 初始化做市商挂单 |
| `execute_all(arena_commands)` | 执行所有竞技场的决策 |
| `get_all_depths()` | 获取所有竞技场的订单簿深度 |
| `shutdown()` | 关闭所有 Worker 并清理共享内存 |

**使用示例：**
```python
from src.training.arena import ArenaExecuteWorkerPoolShm, ArenaExecuteData

# 创建共享内存版 Worker 池
pool = ArenaExecuteWorkerPoolShm(
    num_workers=4,
    arena_ids=[0, 1, 2, 3, 4, 5, 6, 7],
    config=config,
)

# 使用上下文管理器（自动清理共享内存）
with pool:
    # 重置所有竞技场
    pool.reset_all(initial_price=100.0, fee_rates={...})

    # 初始化做市商
    results = pool.init_market_makers({
        0: [(mm_id, bid_orders, ask_orders), ...],
        ...
    })

    # 每个 tick 执行
    results = pool.execute_all({
        0: ArenaExecuteData(
            liquidated_agents=[...],
            decisions=[...],
            mm_decisions=[...],
        ),
        ...
    })
```

**注意事项：**
- Worker 进程使用 `ready_event` 通知主进程就绪
- shutdown 时会自动 unlink 共享内存
- 使用上下文管理器可确保资源正确清理

---

### ParallelArenaTrainer (parallel_arena_trainer.py)

多竞技场并行推理训练器，核心特性是将多个竞技场的神经网络推理合并成一个批量操作。

**设计理念：**
1. **神经网络共享**：所有竞技场使用同一套 `BatchNetworkCache`，进化后统一更新
2. **账户状态独立**：每个竞技场维护独立的 `ArenaState`，包含所有 Agent 的账户状态
3. **批量推理合并**：N 个竞技场 × M 个 Agent 的推理合并成单次 OpenMP 并行操作
4. **订单簿独立**：每个竞技场有独立的 `MatchingEngine` 和 `OrderBook`

**核心流程：**
1. 初始化：创建共享种群、N 个独立竞技场状态、共享网络缓存、进化 Worker 池、Execute Worker 池（可选）
2. 训练循环：
   a. 重置所有竞技场（及 Execute Worker 池的订单簿）
   b. 同步推进所有竞技场的 tick（批量推理 + 并行/串行执行）
   c. 汇总适应度
   d. 执行 NEAT 进化
   e. 更新网络缓存和 Agent 状态

**类定义：**
```python
@dataclass
class MultiArenaConfig:
    """多竞技场配置"""
    num_arenas: int = 2
    episodes_per_arena: int = 50
    use_shared_memory_ipc: bool = True  # 默认启用共享内存 IPC（性能提升约 21%）


class ParallelArenaTrainer:
    """多竞技场并行推理训练器

    Attributes:
        config: 全局配置
        multi_config: 多竞技场配置
        populations: 共享的种群（神经网络）
        arena_states: N 个独立的竞技场状态
        network_caches: 共享的网络缓存
        evolution_worker_pool: 进化 Worker 池
        _execute_worker_pool: Execute Worker 池（可选，禁用鲶鱼时启用）
        _use_execute_workers: 是否使用 Execute Worker 池
        generation: 当前代数
        total_episodes: 总 episode 数
    """
```

**主要方法：**

| 方法 | 描述 |
|------|------|
| `setup()` | 初始化：创建种群、竞技场状态、网络缓存、进化 Worker 池、Execute Worker 池 |
| `run_round()` | 运行一轮训练（所有竞技场的所有 episode + 进化） |
| `run_tick_all_arenas()` | 并行执行所有竞技场的一个 tick（支持 Worker 池并行执行） |
| `_batch_inference_all_arenas()` | 批量推理所有竞技场的所有 Agent |
| `_run_episode_all_arenas()` | 运行所有竞技场的一个 episode |
| `_collect_fitness_all_arenas()` | 收集并汇总所有竞技场的适应度 |
| `_refresh_agent_states()` | 进化后刷新所有竞技场的 Agent 账户状态副本 |
| `_create_execute_worker_pool()` | 创建 Execute Worker 池 |
| `_collect_fee_rates()` | 收集所有 Agent 的费率 |
| `_prepare_mm_init_orders()` | 准备做市商初始化订单 |
| `_execute_with_worker_pool()` | 使用 Worker 池执行决策 |
| `_process_worker_results()` | 处理 Worker 返回的执行结果 |
| `train(num_rounds, checkpoint_callback, progress_callback)` | 主训练循环 |
| `save_checkpoint(path)` | 保存检查点 |
| `load_checkpoint(path)` | 加载检查点 |
| `stop()` | 停止训练并清理资源（包括关闭 Execute Worker 池） |

**账户完全独立设计：**

各竞技场之间只共享 Agent 的基因（神经网络），账户状态完全独立：
- **资金/余额**：每个竞技场维护独立的 `balance`
- **持仓**：每个竞技场维护独立的 `position_quantity` 和 `position_avg_price`
- **挂单 ID**：做市商的 `bid_order_ids` 和 `ask_order_ids` 各竞技场独立
- **行情**：每个竞技场有独立的订单簿和价格走势

订单数量和倾斜因子计算使用独立的辅助函数，直接基于 `AgentAccountState` 计算，不依赖 `Agent.account`：
- `calculate_order_quantity_from_state(state, price, ratio, is_buy, ref_price)` - 基于 AgentAccountState 计算订单数量
- `calculate_skew_factor_from_state(state, mid_price)` - 基于 AgentAccountState 计算做市商仓位倾斜因子

**tick 执行流程（run_tick_all_arenas）：**
```python
# 阶段1: 准备（串行）
for arena in arena_states:
    handle_liquidations(arena)  # 强平检查
    catfish_action(arena)       # 鲶鱼行动（在 Agent 之前）
    market_state = compute_market_state(arena)
    active_agents = get_active_agents(arena)

# 阶段2: 批量推理（并行）- 一次性推理所有竞技场的所有 Agent
all_decisions = _batch_inference_all_arenas(market_states, active_agents)

# 阶段3: 执行
if _execute_worker_pool is not None:
    # 使用 Worker 池并行执行
    results = _execute_with_worker_pool(all_decisions)
    _process_worker_results(results)
else:
    # 串行执行
    for arena in arena_states:
        execute_trades(arena, all_decisions[arena.arena_id])
```

**Execute Worker 池集成：**

当 `_use_execute_workers=True` 且鲶鱼禁用时，Execute 阶段会使用 `ArenaExecuteWorkerPool` 并行化：

1. **setup() 阶段**：如果条件满足，创建 Execute Worker 池
2. **_reset_all_arenas() 阶段**：重置 Worker 池中各竞技场的订单簿
3. **_init_market_all_arenas() 阶段**：使用 Worker 池执行做市商初始化挂单
4. **run_tick_all_arenas() 阶段3**：
   - 将决策数据转换为 `ArenaExecuteData` 格式
   - 调用 `_execute_worker_pool.execute_all()` 并行执行
   - 处理返回结果，更新主进程中的 `AgentAccountState`
5. **stop() 阶段**：关闭 Worker 池

**开关控制：**
- `_use_execute_workers: bool = True`：默认启用
- 当鲶鱼启用时自动回退到串行执行（鲶鱼需要在 Agent 之前行动，与 Worker 池不兼容）

**鲶鱼行动流程（_catfish_action_for_arena）：**
1. 调用 `CatfishAccountState.decide(tick, price_history)` 获取决策
2. 计算下单数量（吃掉前3档）
3. 创建市价单并执行
4. 更新鲶鱼账户和 maker 账户状态
5. 记录行动时间

**独立随机性：**
每个竞技场的趋势创造者鲶鱼在 `reset()` 时独立随机选择方向，确保不同竞技场产生不同的行情走势。

**批量推理合并（_batch_inference_all_arenas）：**
- 使用 `AgentStateAdapter` 将 `AgentAccountState` 适配为 Agent-like 接口
- 按 AgentType 分组（跨所有竞技场）
- 使用 `_network_index_map` 获取每个 Agent 在其种群中的网络索引
- 调用 `BatchNetworkCache.decide_multi_arena()` 一次性处理所有竞技场的推理
- 将结果重组为 `dict[arena_id, list[decision]]`

**辅助方法：**
- `_build_network_index_map()` - 构建 Agent ID 到网络索引的映射表
- `_get_network_index(agent_type, agent_id)` - 获取 Agent 在其种群中的网络索引
- `_parse_market_maker_output(agent_state, output, mid_price, tick_size)` - 解析做市商神经网络输出（41个值）为订单列表，使用 AgentAccountState
- `_parse_non_mm_output(agent_state, outputs, mid_price, tick_size)` - 解析非做市商神经网络输出（8个值），使用 AgentAccountState
- `_convert_retail_result(agent_state, action_type_int, side_int, price, quantity, mid_price)` - 转换散户/高级散户/庄家的决策结果，使用 AgentAccountState
- `_execute_mm_action_in_arena(arena, agent_state, params)` - 在竞技场中执行做市商动作（不依赖 Agent 对象）
- `_execute_non_mm_action_in_arena(arena, agent_state, action, params)` - 在竞技场中执行非做市商动作（不依赖 Agent 对象）
- `_update_trade_accounts(arena, agent_state, trades)` - 更新成交相关的账户状态（taker 和 maker）
- `_cancel_agent_orders_in_arena(arena, agent_state)` - 撤销 Agent 在竞技场中的挂单
- `_serial_inference_for_arena(arena_idx, market_state, adapters)` - 串行推理回退方案，同步 AgentAccountState 到 Agent.account 后推理
- `_sync_state_to_agent(agent, state)` - 临时同步 AgentAccountState 到 Agent.account（用于回退推理路径）
- `_catfish_action_for_arena(arena)` - 鲶鱼在竞技场中的行动，实现三种鲶鱼的决策和执行逻辑
- `_calculate_catfish_quantity(orderbook, direction)` - 计算鲶鱼下单数量（吃掉前3档）
- `_build_decisions_array_from_cache(arena_idx)` - 从缓存的推理结果构建 decisions_array（NumPy 数组格式）

**NumPy 数组格式决策数据传输优化：**

`_batch_inference_all_arenas_direct` 方法现在支持 NumPy 数组格式的决策数据传输，用于优化 Worker 池执行阶段的数据传输：

1. **推理阶段**：对于非做市商类型（RETAIL, RETAIL_PRO, WHALE），调用 `BatchNetworkCache.decide_multi_arena_direct(return_array=True)` 获取 NumPy 数组格式的决策结果
2. **缓存结构**：`_last_inference_arrays: dict[AgentType, dict[arena_idx, (agent_ids, decisions_array)]]`
   - `agent_ids`: shape `(num_agents,)`，与 decisions_array 行对应的 agent_id
   - `decisions_array`: shape `(num_agents, 4)`，列顺序 `[action_type, side, price, quantity]`
3. **执行阶段**：`_build_decisions_array_from_cache(arena_idx)` 将各 AgentType 的缓存数组合并，添加 agent_id 列，构建 `ArenaExecuteData.decisions_array`
   - 输出 shape `(N, 5)`，列顺序 `[agent_id, action_type, side, price, quantity]`
   - 自动过滤掉 HOLD 动作（action_type == 0）

**优化效果**：
- 避免在执行阶段从 list 重新构建 NumPy 数组
- 减少 Python 对象创建开销
- 为后续共享内存优化做准备

**做市商输出解析（_parse_market_maker_output）：**
神经网络输出结构（共 41 个值），与 `MarketMakerAgent.decide()` 保持一致：
- 输出[0-9]: 买单1-10的价格偏移（-1到1，映射到1-100 ticks）
- 输出[10-19]: 买单1-10的数量权重（-1到1，映射到0-1）
- 输出[20-29]: 卖单1-10的价格偏移（-1到1，映射到1-100 ticks）
- 输出[30-39]: 卖单1-10的数量权重（-1到1，映射到0-1）
- 输出[40]: 总下单比例基准（-1到1，映射到0.01-1）

**重要实现细节：**

Cython 批量推理 (`BatchNetworkCache.decide_multi_arena_direct`) 返回的数组格式为：
- 列 0: action_type (int)
- 列 1: side (int)
- 列 2: price (float)
- 列 3: **quantity_ratio** (float 0-1)，不是实际订单数量！

调用方需要将 `quantity_ratio` 传给 `calculate_order_quantity_from_state()` 函数来计算实际订单数量。

**使用示例：**
```python
from src.training.arena import ParallelArenaTrainer, MultiArenaConfig

# 方式1：使用上下文管理器
multi_config = MultiArenaConfig(num_arenas=2, episodes_per_arena=50)
with ParallelArenaTrainer(config, multi_config) as trainer:
    trainer.train(
        num_rounds=100,
        checkpoint_callback=lambda gen: print(f"Gen {gen}"),
        progress_callback=lambda stats: print(stats)
    )

# 方式2：手动管理
trainer = ParallelArenaTrainer(config, multi_config)
trainer.setup()
try:
    for _ in range(100):
        stats = trainer.run_round()
        if trainer.generation % 10 == 0:
            trainer.save_checkpoint(f"checkpoints/gen_{trainer.generation}.pkl")
finally:
    trainer.stop()
```

**启动脚本：**
```bash
# 默认训练
python scripts/train_parallel_arena.py --rounds 100

# 自定义参数
python scripts/train_parallel_arena.py --num-arenas 8 --episodes-per-arena 5 --rounds 200

# 从检查点恢复
python scripts/train_parallel_arena.py --resume checkpoints/parallel_arena_gen_50.pkl
```

**检查点格式：**
```python
{
    "generation": int,
    "populations": {
        AgentType.RETAIL: {
            "is_sub_population_manager": True,
            "sub_population_count": int,
            "sub_populations": [
                {"generation": int, "neat_pop": neat.Population},
                ...
            ]
        },
        AgentType.RETAIL_PRO: {
            "generation": int,
            "neat_pop": neat.Population,
        },
        ...
    }
}
```

**性能优化：**
1. **批量推理合并**：N 个竞技场 × M 个 Agent 合并成单次 OpenMP 并行操作
2. **并行进化**：使用 MultiPopulationWorkerPool 在多个进程中并行执行 NEAT 进化
3. **网络参数传输**：只传输网络参数而非完整基因组，减少序列化开销
4. **内存管理**：每轮训练后进行垃圾回收和 malloc_trim
5. **Checkpoint 体积优化**：使用 gzip 压缩保存检查点文件
